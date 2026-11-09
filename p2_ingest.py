# %% [markdown]
# ---
# ## Task 1 — PDF processing pipeline and summarisation
# ### 1.1 Document ingestion
# The PDF is downloaded from imf.org (with an upload fallback in Colab). `pypdf` extracts every page **twice**:
# * **`plain` mode** follows the PDF's internal content-stream order. That is good for running prose.
# * **`layout` mode** rebuilds the *visual* positions of the text. It is essential for tables. On p.36 of this report, Tables 5a (Commercial Banks) and 5b (Credit Unions) share a page, and the content stream emits the credit-union block *first*. Plain-mode parsing therefore pins credit-union NPLs on banks. Layout mode keeps each block under its correct title.
#
# Extracted pages are cached to `data/pages.json`, keyed by the PDF's SHA-256.

# %%
def ensure_pdf(path: Path, url: str) -> Path:
    """Return a local copy of the IMF report: existing file → download → Colab upload."""
    if path.exists() and path.stat().st_size > 10_000:
        return path
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (academic-research-notebook)"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
        if data[:4] == b"%PDF":
            path.write_bytes(data)
            print(f"Downloaded {len(data)/1e6:.1f} MB from imf.org")
            return path
        print("Download did not return a PDF (got", data[:20], ")")
    except Exception as exc:                                  # noqa: BLE001
        print("Automatic download failed:", exc)
    if IN_COLAB:
        from google.colab import files
        print(f"Please upload the IMF PDF ({CONFIG['imf_country_report_no']}).")
        uploaded = files.upload()
        name, content = next(iter(uploaded.items()))
        path.write_bytes(content)
        return path
    raise FileNotFoundError(f"Place the report at {path} (source: {url})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract plain + layout text per page, caching the result keyed by the PDF hash."""
    digest = sha256_file(pdf_path)
    cache = DATA_DIR / "pages.json"
    if cache.exists():
        rec = json.loads(cache.read_text(encoding="utf-8"))
        if rec.get("sha256") == digest:
            return rec["pages"]
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        plain = page.extract_text() or ""
        try:
            layout = page.extract_text(extraction_mode="layout") or plain
        except Exception:                                     # noqa: BLE001 - very old pypdf / odd page
            layout = plain
        pages.append({"page": i, "plain": plain, "layout": layout})
    cache.write_text(json.dumps({"sha256": digest, "pages": pages}, ensure_ascii=False), encoding="utf-8")
    return pages


PDF_PATH = ensure_pdf(PDF_PATH, CONFIG["pdf_url"])
RAW_PAGES = extract_pages(PDF_PATH)
print(f"{len(RAW_PAGES)} pages | {sum(len(p['plain']) for p in RAW_PAGES):,} characters (plain mode)")
print("--- first 400 characters of page 2 (raw) ---\n", RAW_PAGES[1]["plain"][:400])

# %% [markdown]
# ### 1.2 Cleaning: unicode, running headers/footers, artefacts
# IMF reports produce a few typical kinds of noise, each handled by a separate function:
#
# | Noise | Example from this report | Fix |
# |---|---|---|
# | Unicode fractions | `43½ percentage point`, `9¾ percent` | → `43.5`, `9.75` (otherwise the LLM sees "43" and "½") |
# | Ligatures, typographic dashes and quotes | `ﬁ`, `–`, `’` | NFKC normalisation + ASCII mapping |
# | Running headers/footers | `DOMINICA`, `INTERNATIONAL MONETARY FUND 13` | **learned** from line frequency at page edges, not hard-coded |
# | Table-of-contents leaders | `CONTEXT_____ 4` | dropped |
# | Hard line wraps | sentences split mid-line | paragraph reflow (numbered IMF paragraphs start new blocks) |
#
# Lines with three or more numbers (table rows, year headers) are never treated as boilerplate, so the table parser can still see them.

# %%
FRACTIONS = {"½": ".5", "¼": ".25", "¾": ".75", "⅓": ".33", "⅔": ".67", "⅛": ".125"}
CHAR_MAP = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
            "−": "-", "‐": "-", "‑": "-", " ": " ", "…": "...", "•": "•",
            "­": ""}
NUM_TOKEN = re.compile(r"^\(?-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?%?$|^\(?-?\d+(?:\.\d+)?\)?%?$")
MISSING_TOKEN = re.compile(r"^(?:n\.?a\.?|\.\.\.|…|-{1,2}|—)$", re.IGNORECASE)


def normalise_text(text: str) -> str:
    for frac, dec in FRACTIONS.items():              # must run BEFORE NFKC (NFKC turns ½ into 1⁄2)
        text = re.sub(rf"(\d){frac}", lambda m, d=dec: m.group(1) + d, text)
        text = text.replace(frac, "0" + dec)
    text = unicodedata.normalize("NFKC", text)
    for src, dst in CHAR_MAP.items():
        text = text.replace(src, dst)
    return text


def n_numbers(line: str) -> int:
    return sum(bool(NUM_TOKEN.match(t)) for t in line.split())


def is_table_row(line: str) -> bool:
    toks = line.split()
    nums = sum(bool(NUM_TOKEN.match(t) or MISSING_TOKEN.match(t)) for t in toks)
    return nums >= 4 and nums / max(1, len(toks)) >= 0.4


def boilerplate_key(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip()))


def learn_boilerplate(pages_lines: List[List[str]], edge: int = 2, min_share: float = 0.2) -> set:
    """Lines that recur in the first/last `edge` lines of >= `min_share` of pages are running headers/footers."""
    counts = Counter()
    for lines in pages_lines:
        nonempty = [l for l in lines if l.strip()]
        for l in set(nonempty[:edge] + nonempty[-edge:]):
            ls = l.strip()
            if n_numbers(ls) < 3 and len(ls) < 80 and len(ls.split()) <= 8 and not ls.endswith("."):
                counts[boilerplate_key(l)] += 1
    n = max(1, len(pages_lines))
    return {k for k, v in counts.items() if v / n >= min_share}


ARTEFACT_PATTERNS = [
    re.compile(r"^\s*\d{1,3}\s*$"),                       # bare page numbers
    re.compile(r"_{4,}"),                                 # TOC leader lines
    re.compile(r"^\s*©\s*\d{4}\s+International Monetary Fund", re.I),
]


def clean_page_lines(text: str, boilerplate: set) -> List[str]:
    out = []
    for line in normalise_text(text).splitlines():
        s = re.sub(r"[ \t]+", " ", line).strip()
        if not s or boilerplate_key(s) in boilerplate or any(p.search(s) for p in ARTEFACT_PATTERNS):
            continue
        out.append(s)
    return out


def clean_layout_lines(text: str, boilerplate: set) -> List[str]:
    """Layout text keeps its column spacing (collapsed to max 3 spaces) so table rows stay intact."""
    out = []
    for line in normalise_text(text).splitlines():
        s = re.sub(r" {3,}", "   ", line.rstrip())
        if not s.strip() or boilerplate_key(s) in boilerplate or any(p.search(s) for p in ARTEFACT_PATTERNS):
            continue
        out.append(s.strip())
    return out


_plain_lines = [normalise_text(p["plain"]).splitlines() for p in RAW_PAGES]
_layout_lines = [[re.sub(r"\s+", " ", l) for l in normalise_text(p["layout"]).splitlines()] for p in RAW_PAGES]
BOILERPLATE = learn_boilerplate(_plain_lines) | learn_boilerplate(_layout_lines)   # layout mode merges header lines
PAGES = [{"page": p["page"],
          "lines": clean_page_lines(p["plain"], BOILERPLATE),
          "layout_lines": clean_layout_lines(p["layout"], BOILERPLATE)} for p in RAW_PAGES]

print("Learned running headers/footers:", sorted(BOILERPLATE))
raw_chars = sum(len(p["plain"]) for p in RAW_PAGES)
clean_chars = sum(len(" ".join(p["lines"])) for p in PAGES)
print(f"Characters: raw {raw_chars:,} → cleaned {clean_chars:,} ({1 - clean_chars / raw_chars:.1%} removed)")
print("--- page 9 after cleaning (first 8 lines) ---")
print("\n".join(PAGES[min(8, len(PAGES) - 1)]["lines"][:8]))

# %% [markdown]
# ### 1.3 Document structure: parts and sections
# An Article IV *package* bundles several documents. Parts are detected from markers near the top of each page and are only allowed to move **forward** (each later cover page also repeats "STAFF REPORT FOR THE 2024 ARTICLE IV CONSULTATION", so a naive search would jump back):
#
# `Press Release → Staff Report → Informational Annex → Debt Sustainability Analysis → Executive Director Statement`
#
# Inside the Staff Report, chapters are detected from layout cues instead of a hard-coded list: stand-alone ALL-CAPS headings (`RECENT DEVELOPMENTS`), `Annex I.` headings, and the start of the figures/tables block. The shorter parts are summarised as single units.

# %%
PART_MARKERS = [  # (part name, regex searched in the first ~600 characters of a page) – checked latest-first
    ("Executive Director Statement", re.compile(r"(?m)^Statement by ")),
    ("Debt Sustainability Analysis", re.compile(r"DEBT SUSTAINABILITY\s+ANALYSIS")),
    ("Informational Annex", re.compile(r"INFORMATIONAL\s+ANNEX")),
    ("Staff Report", re.compile(r"STAFF REPORT FOR THE \d{4}\s+ARTICLE IV")),
    ("Press Release", re.compile(r"FOR IMMEDIATE RELEASE|^PR\s?\d{2}/\d+", re.M)),
]
PART_ORDER = ["Cover", "Press Release", "Staff Report", "Informational Annex",
              "Debt Sustainability Analysis", "Executive Director Statement"]
WHOLE_PARTS = {"Press Release", "Informational Annex", "Debt Sustainability Analysis",
               "Executive Director Statement"}
HEADING_STOPLIST = {"CONTENTS", "FIGURES", "TABLES", "BOXES", "ANNEXES", "APPENDICES", "APPENDIX",
                    "FOR IMMEDIATE RELEASE", "APPROVED BY", "ANNEX", "SOURCES", "NOTES"}
UPPER_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,&'\-:()/]{3,90}$")
ANNEX_HEADING = re.compile(r"^Annex\s+([IVXL]+)\.\s+(?!(?:Figure|Table|Box)\b)(\S.+)$")
FIG_TAB_START = re.compile(r"^(?:Figure|Table)\s+1[a-z]?\.\s+[A-Z][\w' ]+:")
TITLE_LINE = re.compile(r"^(?:Annex\s+[IVXL]+\.\s+)?(?:Table|Figure|Box)\s+\d+[a-z]?\.\s+\S")


def detect_part(page_text_head: str, current: str) -> str:
    for name, rx in PART_MARKERS:
        if rx.search(page_text_head) and PART_ORDER.index(name) > PART_ORDER.index(current):
            return name
    return current


def is_upper_heading(line: str) -> bool:
    words = line.split()
    if not UPPER_HEADING.match(line) or line in HEADING_STOPLIST or len(words) > 10:
        return False
    if sum(c.isalpha() for c in line) < 4 or (len(words) > 1 and len(set(words)) == 1):
        return False
    if any(rx.search(line) for _, rx in PART_MARKERS):
        return False
    return n_numbers(line) == 0


def page_is_tabular(lines: List[str]) -> bool:
    """Figure/table pages: dominated by numeric rows or short chart fragments."""
    if not lines:
        return True
    rows = sum(is_table_row(l) for l in lines)
    short = sum(len(l) < 45 for l in lines)
    has_title = any(TITLE_LINE.match(l) for l in lines)
    return rows / len(lines) > 0.3 or (has_title and short / len(lines) > 0.55)


def is_sentence_like(line: str) -> bool:
    return len(line.split()) >= 6 and bool(re.search(r"[a-z]", line)) and not is_table_row(line)


@dataclass
class Section:
    sid: str
    part: str
    title: str
    start_page: int
    end_page: int
    prose: List[Tuple[int, str]] = field(default_factory=list)      # (page, line)
    tables: List[Tuple[int, str]] = field(default_factory=list)     # (page, layout line)
    subsections: List[str] = field(default_factory=list)

    @property
    def prose_text(self) -> str:
        return "\n\n".join(p for _, p in reflow(self.prose))

    @property
    def n_words(self) -> int:
        return len(self.prose_text.split())


def reflow(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Join hard-wrapped lines into paragraphs; IMF numbered paragraphs, bullets and headings start new ones."""
    paras, buf, buf_page = [], "", None
    new_block = re.compile(r"^(?:\d{1,2}\.\s+[A-Z]|[•▪]\s*|\(?[a-z]\)\s|[A-H]\.\s+[A-Z])")
    for page, line in lines:
        if new_block.match(line) or is_upper_heading(line) or TITLE_LINE.match(line):
            if buf:
                paras.append((buf_page, buf))
            buf, buf_page = line, page
            continue
        if not buf:
            buf, buf_page = line, page
        elif buf.endswith("-") and line[:1].islower():
            buf += line                                   # keep the hyphen: 'climate-' + 'resilient'
        else:
            buf += " " + line
    if buf:
        paras.append((buf_page, buf))
    return paras


def build_sections(pages: List[Dict[str, Any]]) -> List[Section]:
    sections: List[Section] = []
    part, cur = "Cover", None

    def open_section(title: str, page: int, part_name: str) -> Section:
        s = Section(sid=f"S{len(sections) + 1:02d}", part=part_name, title=title, start_page=page, end_page=page)
        sections.append(s)
        return s

    for pg in pages:
        lines, page_no = pg["lines"], pg["page"]
        new_part = detect_part("\n".join(lines)[:600], part)
        if new_part != part or cur is None:
            part = new_part
            cur = open_section(part, page_no, part)
        tabular = page_is_tabular(lines)
        if tabular:
            cur_tables = pg["layout_lines"]
            if part == "Staff Report" and not cur.title.startswith("Figures") and \
                    any(FIG_TAB_START.match(l) for l in lines):
                cur = open_section("Figures and Statistical Tables", page_no, part)
            cur.tables.extend((page_no, l) for l in cur_tables)
            cur.prose.extend((page_no, l) for l in lines if is_sentence_like(l))   # keep chart captions
            cur.end_page = page_no
            continue
        i = 0
        while i < len(lines):
            line = lines[i]
            if part not in WHOLE_PARTS:
                m = ANNEX_HEADING.match(line)
                if m:
                    if not cur.title.startswith(f"Annex {m.group(1)}."):
                        cur = open_section(f"Annex {m.group(1)}. {m.group(2).rstrip('0123456789')}", page_no, part)
                    i += 1
                    continue
                if is_upper_heading(line):
                    title = line
                    while i + 1 < len(lines) and is_upper_heading(lines[i + 1]):   # two-line headings
                        i += 1
                        title += " " + lines[i]
                    cur = open_section(title.title(), page_no, part)
                    i += 1
                    continue
                if part == "Staff Report" and FIG_TAB_START.match(line) and not cur.title.startswith("Figures"):
                    cur = open_section("Figures and Statistical Tables", page_no, part)
            if re.match(r"^[A-H]\.\s+[A-Z][a-z]", line) and len(line) < 110:
                cur.subsections.append(line)
            if is_table_row(line):
                cur.tables.append((page_no, line))
            else:
                cur.prose.append((page_no, line))
            cur.end_page = page_no
            i += 1
    # merge tiny fragments (e.g. a heading immediately followed by another) into the next section
    merged: List[Section] = []
    for s in sections:
        if merged and merged[-1].n_words < 40 and not merged[-1].tables and merged[-1].part == s.part \
                and merged[-1].part not in WHOLE_PARTS:
            prev = merged.pop()
            s.prose = prev.prose + s.prose
            s.start_page = prev.start_page
        merged.append(s)
    for k, s in enumerate(merged, start=1):
        s.sid = f"S{k:02d}"
    return merged


SECTIONS = build_sections(PAGES)
section_df = pd.DataFrame([{"sid": s.sid, "part": s.part, "title": s.title, "pages": f"{s.start_page}-{s.end_page}",
                            "prose_words": s.n_words, "table_lines": len(s.tables),
                            "subsections": "; ".join(s.subsections[:4])} for s in SECTIONS])
display(section_df)
save_json([{**asdict(s), "prose_text": s.prose_text} for s in SECTIONS], "sections.json")
