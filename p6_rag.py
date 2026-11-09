# %% [markdown]
# ---
# ## Task 3 — RAG grounding and retrieval
# The FAISS index from §1.4 now answers factual policy questions about the report. Each question runs under three conditions:
# 1. **Closed-book** (no retrieval): the model answers from its parametric memory. This is the *hallucination baseline*.
# 2. **Single-query RAG**: top-k = 5 chunks for the original question.
# 3. **Multi-Query RAG** (*extension b*): the LLM writes 3 paraphrases, each query retrieves top-k chunks, and the ranked lists are fused with **Reciprocal Rank Fusion** (Cormack et al., 2009) into a top-8 context.
#
# Grounded answers must cite chunk ids, and the cited ids are checked against the context the model was actually given. The **judge scores all three answers against one shared evidence pool** (the union of both retrievals), so that faithfulness and completeness are comparable across conditions. A cheap **retrieval hit test** (do the expected key terms appear in the retrieved text?) checks retrieval quality separately from generation quality.

# %%
RAG_QUESTIONS = [
    {"q": "What does the IMF recommend on bank loan-loss provisioning and capital for credit unions?",
     "expect": ["provision", "credit union", "capital"]},
    {"q": "Which shocks and assumptions are used in the debt sustainability stress tests, such as a natural disaster scenario?",
     "expect": ["stress test", "natural disaster"]},
    {"q": "What is the outlook for inflation and how quickly is it expected to converge to about 2 percent?",
     "expect": ["inflation", "2 percent"]},
    {"q": "How much do fiscal accounts rely on Citizenship by Investment revenues and what risk does that create?",
     "expect": ["CBI", "revenue"]},
    {"q": "What public debt target applies to Dominica and by when is it expected to be reached?",
     "expect": ["60 percent", "debt"]},
]

RAG_ANSWER_PROMPT = """Answer the QUESTION about {country} using ONLY the numbered CONTEXT passages.
- Cite the supporting passage ids in square brackets after each claim, e.g. [C012].
- If the context does not contain the answer, reply exactly: "Not stated in the retrieved context."
- Max 120 words. Keep figures and years exactly as written.

QUESTION: {question}

CONTEXT:
{context}"""
CLOSED_BOOK_PROMPT = """Answer the question about the IMF's 2024 Article IV consultation with {country} from your own knowledge (max 120 words).
QUESTION: {question}"""
MULTI_QUERY_PROMPT = """Generate {n} alternative search queries that would retrieve passages answering the question below from an
IMF Article IV report on {country}. Vary vocabulary (technical synonyms, related indicators) while keeping the same intent.
QUESTION: {question}"""
RAG_JUDGE_PROMPT = """Evaluate an ANSWER to a QUESTION about {country} against the EVIDENCE passages from the IMF report.
Score integers 1-5:
- faithfulness: 5 = every claim is supported by the evidence; 3 = one unsupported claim; 1 = mostly unsupported or fabricated.
- completeness: 5 = covers everything the evidence says that answers the question; 3 = misses an important element; 1 = misses most.
- relevance: 5 = directly answers the question; 1 = off-topic.
List unsupported claims verbatim.

QUESTION: {question}
ANSWER: {answer}
EVIDENCE:
{evidence}"""
PROMPTS.update({"RAG_ANSWER_PROMPT": RAG_ANSWER_PROMPT, "CLOSED_BOOK_PROMPT": CLOSED_BOOK_PROMPT,
                "MULTI_QUERY_PROMPT": MULTI_QUERY_PROMPT, "RAG_JUDGE_PROMPT": RAG_JUDGE_PROMPT})


class QueryVariants(BaseModel):
    queries: List[str]


class RAGJudgement(BaseModel):
    faithfulness: int
    completeness: int
    relevance: int
    unsupported_claims: List[str] = Field(default_factory=list)
    rationale: str = ""


def rrf_fuse(ranked_lists: List[List[Tuple[Chunk, float]]], k_rrf: int, top_n: int) -> List[Tuple[Chunk, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_d)."""
    scores, by_id = defaultdict(float), {}
    for lst in ranked_lists:
        for rank, (c, _) in enumerate(lst, start=1):
            scores[c.chunk_id] += 1.0 / (k_rrf + rank)
            by_id[c.chunk_id] = c
    best = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [(by_id[i], scores[i]) for i in best]


def retrieval_hit(hits: List[Tuple[Chunk, float]], expect: List[str]) -> float:
    text = " ".join(c.text.lower() for c, _ in hits)
    return sum(e.lower() in text for e in expect) / len(expect)


def cited_ids(answer: str) -> List[str]:
    return re.findall(r"C\d{3}", answer)


def judge_rag(question: str, answer: str, evidence: str, tag: str) -> Optional[RAGJudgement]:
    res = llm.generate(RAG_JUDGE_PROMPT.format(country=C, question=question, answer=answer, evidence=evidence),
                       JUDGE_MODEL, system=SYSTEM_JUDGE, schema=RAGJudgement, tag=tag)
    if not res.ok:
        return None
    j = res.parsed
    for f_ in ("faithfulness", "completeness", "relevance"):
        setattr(j, f_, int(min(5, max(1, getattr(j, f_)))))
    return j


RAG_RESULTS = []
for qi, item in enumerate(RAG_QUESTIONS, start=1):
    q = item["q"]
    single = VINDEX.search(q, k=CONFIG["retrieval_k"])
    mq_res = llm.generate(MULTI_QUERY_PROMPT.format(n=CONFIG["multi_query_n_variants"], country=C, question=q),
                          EXTRACTOR_MODEL, schema=QueryVariants, tag=f"rag{qi}-variants")
    variants = (mq_res.parsed.queries[:CONFIG["multi_query_n_variants"]] if mq_res.ok else [])
    multi = rrf_fuse([VINDEX.search(v, k=CONFIG["retrieval_k"]) for v in [q] + variants],
                     CONFIG["rrf_k"], CONFIG["multi_query_top_k"])
    evidence_pool = format_context(single + multi, max_tokens=12_000)

    conditions = {
        "closed_book": (None, llm.generate(CLOSED_BOOK_PROMPT.format(country=C, question=q), EXTRACTOR_MODEL,
                                           tag=f"rag{qi}-closed")),
        "single_query": (single, llm.generate(RAG_ANSWER_PROMPT.format(country=C, question=q, context=format_context(single)),
                                              EXTRACTOR_MODEL, system=SYSTEM_ANALYST, tag=f"rag{qi}-single")),
        "multi_query": (multi, llm.generate(RAG_ANSWER_PROMPT.format(country=C, question=q, context=format_context(multi)),
                                            EXTRACTOR_MODEL, system=SYSTEM_ANALYST, tag=f"rag{qi}-multi")),
    }
    rec = {"question": q, "variants": variants, "retrieval": {}, "answers": {}}
    for name, (hits, res) in conditions.items():
        answer = res.text.strip()
        given = {c.chunk_id for c, _ in hits} if hits else set()
        cites = cited_ids(answer)
        j = judge_rag(q, answer, evidence_pool, f"rag{qi}-judge-{name}")
        rec["answers"][name] = {
            "answer": answer, "latency_s": round(res.latency_s, 2), "error": res.error,
            "citations": cites, "citation_validity": (sum(c in given for c in cites) / len(cites)) if cites else None,
            "judge": j.model_dump() if j else None}
        if hits:
            rec["retrieval"][name] = {"chunks": [{"chunk_id": c.chunk_id, "section": c.section, "pages":
                                                  f"{c.page_start}-{c.page_end}", "score": round(s, 4)} for c, s in hits],
                                      "hit_rate": retrieval_hit(hits, item["expect"]),
                                      "unique_sections": len({c.sid for c, _ in hits})}
    RAG_RESULTS.append(rec)
    print(f"Q{qi}: hit single={rec['retrieval']['single_query']['hit_rate']:.2f} "
          f"multi={rec['retrieval']['multi_query']['hit_rate']:.2f}")

rag_rows = []
for qi, r in enumerate(RAG_RESULTS, start=1):
    for cond, a in r["answers"].items():
        if a["judge"]:
            rag_rows.append({"q": f"Q{qi}", "condition": cond, **{k: a["judge"][k] for k in
                             ("faithfulness", "completeness", "relevance")},
                             "n_unsupported": len(a["judge"]["unsupported_claims"]),
                             "citation_validity": a["citation_validity"],
                             "hit_rate": r["retrieval"].get(cond, {}).get("hit_rate")})
RAG_DF = pd.DataFrame(rag_rows)
if not RAG_DF.empty:
    display(RAG_DF.groupby("condition")[["faithfulness", "completeness", "relevance", "n_unsupported",
                                         "citation_validity", "hit_rate"]].mean().round(2))
display(Markdown(f"**Example (Q1, single-query RAG):** {RAG_RESULTS[0]['answers']['single_query']['answer']}"))
save_json({"embedding_model": CONFIG["embedding_model"], "index": "faiss.IndexFlatL2 (unit-normalised vectors)",
           "k": CONFIG["retrieval_k"], "multi_query_top_k": CONFIG["multi_query_top_k"], "results": RAG_RESULTS},
          "rag_results.json")

# %% [markdown]
# ---
# ## Extension (a) — Model and system-directive benchmarking
# The *same* indicator-extraction task, on the *same* retrieved context, runs as a 2 × 2 design:
#
# | | strict (`system_instruction` + `response_schema`) | generic (plain user prompt, no schema) |
# |---|---|---|
# | Flash | ✔ | ✔ |
# | Flash-Lite | ✔ | ✔ |
#
# Metrics: **schema compliance** (strict key and type check), **latency** (seconds, first non-cached call), **accuracy vs the human ground truth**, and the **judge support rate**. One fixed judge scores all four cells, so differences come from the extractor configuration and not from the judge.

# %%
BENCH_MODELS = [m for m in dict.fromkeys([EXTRACTOR_MODEL, LITE_MODEL]) if m]
BENCH_ROWS, BENCH_RUNS = [], {}
for model in BENCH_MODELS:
    for style in ("strict", "generic"):
        run = MAIN_RUN if (model == EXTRACTOR_MODEL and style == "strict") else run_indicator_extraction(model, style)
        core, fb = apply_fallbacks(run)
        audit = MAIN_AUDIT if run is MAIN_RUN else audit_indicators(run, f"indicator-audit-{style}-{model}")
        acc = accuracy_vs_truth(core.model_dump())
        BENCH_RUNS[f"{model}|{style}"] = {"run": run, "core": core.model_dump(), "audit": audit, "accuracy": acc}
        BENCH_ROWS.append({"model": model, "style": style, "label": f"{model.replace('gemini-', '')}\n{style}",
                           "schema_compliant": int(run["compliance"]["compliant"]),
                           "fields_populated": sum(core.model_dump()[k] is not None for k in INDICATORS),
                           "accuracy": acc["accuracy"], "judge_support_rate": audit["support_rate"],
                           "judge_overall_1to5": audit["overall_score"], "latency_s": run["latency_s"],
                           "error": run["error"]})
BENCH_DF = pd.DataFrame(BENCH_ROWS)
display(BENCH_DF.drop(columns=["label"]))
save_json({"design": "models x prompt style, same retrieved context, fixed judge", "judge_model": JUDGE_MODEL,
           "results": BENCH_DF.drop(columns=["label"]).to_dict(orient="records"),
           "runs": {k: {"core": v["core"], "audit": v["audit"], "accuracy": v["accuracy"],
                        "compliance": v["run"]["compliance"]} for k, v in BENCH_RUNS.items()}}, "benchmark.json")
