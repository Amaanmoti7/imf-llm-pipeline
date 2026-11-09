# %% [markdown]
# ### 1.4 Token-aware chunking + vector index
# * **Token counting** uses the *embedding model's own tokenizer*. A chunk's token count is then exactly what the embedder sees, and every chunk fits its 512-token window. If that tokenizer is unavailable, `tiktoken` takes over, then a regex estimate.
# * **Prose chunks** are built from whole sentences, target **400 tokens**, with a sentence-aligned **~50-token overlap**. A fact on a chunk boundary therefore appears complete in at least one chunk. Chunks never cross a section boundary, and each carries its section and page range for citation.
# * **Table chunks** come from layout-mode text. When a table spans several chunks, its **title and year header are repeated** at the top of each continuation chunk, so a row like `Regulatory capital ratio 27.51` never loses its "Commercial Banks, 2013–23" context.
#
# The FAISS index is built here, ahead of Task 3, because Task 2's indicator extraction already uses retrieval for grounding. Task 3 then evaluates retrieval on its own.

# %%
class TokenCounter:
    """Embedding-model tokenizer → tiktoken → regex estimate (first that loads wins)."""
    def __init__(self, hf_name: str):
        try:
            from transformers import AutoTokenizer, logging as hf_logging
            hf_logging.set_verbosity_error()
            tok = AutoTokenizer.from_pretrained(hf_name)
            self._fn = lambda s: len(tok.encode(s, add_special_tokens=False))
            self.name = f"{hf_name} tokenizer"
        except Exception:                                     # noqa: BLE001
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                self._fn = lambda s: len(enc.encode(s))
                self.name = "tiktoken cl100k_base"
            except Exception:                                 # noqa: BLE001
                self._fn = lambda s: int(len(re.findall(r"\w+|[^\w\s]", s)) * 1.15)
                self.name = "regex estimate"

    def __call__(self, text: str) -> int:
        return self._fn(text)


count_tokens = TokenCounter(CONFIG["embedding_model"])
print("Token counter:", count_tokens.name)

ABBREVIATIONS = ["e.g.", "i.e.", "U.S.", "U.K.", "No.", "Nos.", "Mr.", "Ms.", "Mrs.", "Dr.", "vs.", "etc.",
                 "Fig.", "Vol.", "St.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec.",
                 "approx.", "incl.", "est."]


def split_sentences(text: str) -> List[str]:
    protected = text
    for a in ABBREVIATIONS:
        protected = protected.replace(a, a.replace(".", "<DOT>"))
    parts = [p.replace("<DOT>", ".").strip()
             for p in re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9•])", protected) if p.strip()]
    merged: List[str] = []
    for p in parts:                       # re-attach IMF paragraph numbers such as "12."
        if merged and len(merged[-1]) <= 4:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


@dataclass
class Chunk:
    chunk_id: str
    sid: str
    section: str
    part: str
    kind: str                 # "prose" | "table"
    page_start: int
    page_end: int
    text: str
    n_tokens: int

    @property
    def embed_text(self) -> str:
        return f"{self.part} | {self.section}: {self.text}"

    @property
    def cite(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        return f"[{self.chunk_id} | {self.section} | {pages}]"


def _split_long(page: int, sent: str, target: int) -> List[Tuple[int, str]]:
    words, out, buf = sent.split(), [], []
    for w in words:
        buf.append(w)
        if count_tokens(" ".join(buf)) >= target:
            out.append((page, " ".join(buf)))
            buf = []
    if buf:
        out.append((page, " ".join(buf)))
    return out


def chunk_units(units: List[Tuple[int, str]], target: int, overlap: int, hard_max: int) -> List[List[Tuple[int, str]]]:
    """Greedy sentence packing with a sentence-aligned (word-level fallback) token overlap."""
    expanded: List[Tuple[int, str]] = []
    for page, s in units:
        expanded.extend(_split_long(page, s, target // 2) if count_tokens(s) > target // 2 else [(page, s)])
    chunks, cur, cur_tok = [], [], 0
    for page, s in expanded:
        t = count_tokens(s)
        if cur and cur_tok + t > target:
            chunks.append(cur)
            tail, tail_tok = [], 0
            for u in reversed(cur):
                ut = count_tokens(u[1])
                if tail_tok + ut > overlap:
                    break
                tail.insert(0, u)
                tail_tok += ut
            if not tail:                                   # last sentence too long → word-level overlap
                last_page, last = cur[-1]
                n_words = max(5, int(overlap / 1.3))
                tail = [(last_page, " ".join(last.split()[-n_words:]))]
                tail_tok = count_tokens(tail[0][1])
            cur, cur_tok = tail, tail_tok
        cur.append((page, s))
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


def chunk_section_prose(sec: Section) -> List[Chunk]:
    units = [(page, sent) for page, para in reflow(sec.prose) for sent in split_sentences(para)]
    groups = chunk_units(units, CONFIG["chunk_target_tokens"], CONFIG["chunk_overlap_tokens"],
                         CONFIG["chunk_hard_max_tokens"])
    out = []
    for g in groups:
        text = " ".join(s for _, s in g)
        n = count_tokens(text)
        if n < CONFIG["chunk_min_tokens"] and out:          # fold a tiny tail into the previous chunk
            prev = out[-1]
            if prev.n_tokens + n <= CONFIG["chunk_hard_max_tokens"]:
                prev.text += " " + text
                prev.n_tokens = count_tokens(prev.text)
                prev.page_end = g[-1][0]
                continue
        out.append(Chunk("", sec.sid, sec.title, sec.part, "prose", g[0][0], g[-1][0], text, n))
    return out


def chunk_section_tables(sec: Section) -> List[Chunk]:
    out, buf, buf_pages, title, header = [], [], [], "", ""

    def flush(force_context: bool):
        nonlocal buf, buf_pages
        body = [l for l in buf if l.strip()]
        if body and any(n_numbers(l) >= 2 for l in body):
            text = "\n".join(body)
            out.append(Chunk("", sec.sid, sec.title, sec.part, "table", min(buf_pages), max(buf_pages),
                             text, count_tokens(text)))
        buf, buf_pages = [], []
        if force_context:                                     # continuation chunk keeps its context
            buf = [l for l in (f"{title} (continued)" if title else "", header) if l]

    for page, line in sec.tables:
        if TITLE_LINE.match(line):
            if buf:
                flush(force_context=False)
            title, header = line, ""
        elif len(years_in(line)) >= 4:
            header = line
        if count_tokens("\n".join(buf + [line])) > CONFIG["chunk_target_tokens"]:
            flush(force_context=True)
        buf.append(line)
        buf_pages.append(page)
    if buf:
        flush(force_context=False)
    return out


def years_in(line: str) -> List[int]:
    return [int(t) for t in re.findall(r"(?<![\d.])(19[89]\d|20[0-5]\d)(?![\d.])", line)]


CHUNKS: List[Chunk] = []
for sec in SECTIONS:
    if sec.part == "Cover":
        continue
    CHUNKS.extend(chunk_section_prose(sec) + chunk_section_tables(sec))
for k, c in enumerate(CHUNKS, start=1):
    c.chunk_id = f"C{k:03d}"
CHUNK_BY_ID = {c.chunk_id: c for c in CHUNKS}

chunk_df = pd.DataFrame([{**asdict(c)} for c in CHUNKS])
print(f"{len(CHUNKS)} chunks ({(chunk_df.kind == 'prose').sum()} prose, {(chunk_df.kind == 'table').sum()} table)")
display(chunk_df.groupby("kind")["n_tokens"].describe().round(1))
over = chunk_df[chunk_df.n_tokens > CONFIG["chunk_hard_max_tokens"]]
print(f"Chunks above the {CONFIG['chunk_hard_max_tokens']}-token hard limit: {len(over)}")

# Overlap check: share of consecutive prose chunks (same section) that share text at the boundary
pairs = [(a, b) for a, b in zip(CHUNKS, CHUNKS[1:]) if a.kind == b.kind == "prose" and a.sid == b.sid]
shared = sum(any(w in b.text[:400] for w in [a.text[-60:].strip()[:40]]) for a, b in pairs)
print(f"Consecutive prose-chunk pairs with boundary overlap: {shared}/{len(pairs)}")
with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
    for c in CHUNKS:
        f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

# %%
class VectorIndex:
    """Dense retrieval: sentence-transformers embeddings (unit-normalised) + FAISS IndexFlatL2.
    On unit vectors, squared L2 distance = 2 − 2·cosine, so L2 ranking equals cosine ranking."""

    def __init__(self, model_name: str, query_prefix: str = ""):
        self.model = SentenceTransformer(model_name)
        self.model_name, self.query_prefix = model_name, query_prefix
        self.index, self.chunks, self.emb = None, [], None

    def build(self, chunks: List[Chunk]) -> "VectorIndex":
        texts = [c.embed_text for c in chunks]
        key = hashlib.sha256((self.model_name + "\x00".join(texts)).encode("utf-8")).hexdigest()[:20]
        path = CACHE_DIR / "embeddings" / f"{key}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            emb = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=False).astype("float32")
            np.save(path, emb)
        self.index = faiss.IndexFlatL2(emb.shape[1])
        self.index.add(emb)
        self.chunks, self.emb = chunks, emb
        return self

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True).astype("float32")

    def search(self, query: str, k: int = 5, kind: Optional[str] = None,
               parts: Optional[Tuple[str, ...]] = None) -> List[Tuple[Chunk, float]]:
        """Return (chunk, cosine similarity) pairs, optionally filtered by chunk kind / document part."""
        n_probe = len(self.chunks) if (kind or parts) else min(k, len(self.chunks))
        dist, idx = self.index.search(self.encode_query(query), n_probe)
        hits = []
        for d, i in zip(dist[0], idx[0]):
            if i < 0:
                continue
            c = self.chunks[i]
            if (kind and c.kind != kind) or (parts and c.part not in parts):
                continue
            hits.append((c, float(1 - d / 2)))
            if len(hits) == k:
                break
        return hits


def format_context(hits: List[Tuple[Chunk, float]], max_tokens: int = 14_000) -> str:
    """Numbered, citable context block for grounded prompts (deduplicated, page-ordered)."""
    seen, blocks, total = set(), [], 0
    for c, _ in sorted(hits, key=lambda h: (h[0].page_start, h[0].chunk_id)):
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        total += c.n_tokens
        if total > max_tokens:
            break
        blocks.append(f"{c.cite} ({c.kind})\n{c.text}")
    return "\n\n".join(blocks)


VINDEX = VectorIndex(CONFIG["embedding_model"], CONFIG["embedding_query_prefix"]).build(CHUNKS)
print(f"FAISS IndexFlatL2: {VINDEX.index.ntotal} vectors × {VINDEX.emb.shape[1]} dims")
for c, s in VINDEX.search("bank nonperforming loans and provisioning", k=3):
    print(f"  {s:.3f} {c.cite} {c.text[:110]}…")
