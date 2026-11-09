# %% [markdown]
# ### 1.5 Prompt library and output schemas
# All prompts are kept in one place and saved to `outputs/prompts.json` so the report can quote them exactly. Design choices:
# * **Role + grounding rule in the system instruction.** "Use only the provided text", with an explicit guard against confusing **Dominica** with **the Dominican Republic**. This report itself writes "the Dominican economy" when it means Dominica, which makes the mix-up easy.
# * **Quantified constraints.** Word limits and bullet counts are stated as numbers, then **verified in code**. The LLM's claim to have complied is never trusted.
# * **Evidence fields.** Structured outputs must carry a verbatim `evidence_quote`, which is fuzzy-matched against the source to catch fabricated quotes.
# * **Judge rubric with anchors.** Each 1–5 score has a defined 1/3/5 meaning, which cuts down score inflation and makes judge runs comparable.

# %%
C = CONFIG["country"]
SYSTEM_ANALYST = (
    f"You are a senior IMF macro-financial analyst summarising an IMF Article IV report on {C}. "
    "Work strictly from the source text supplied in the prompt: never use outside knowledge, never guess, "
    f"and never confuse {C} with {CONFIG['confusable_country']} (the report may say 'the Dominican economy' "
    f"when it means {C}). Reproduce numbers, units and years exactly as written."
)
SYSTEM_JUDGE = (
    "You are an independent, strict evaluator auditing AI-generated text against a source document. "
    "You did not write the candidate. Judge only what the source supports; do not reward fluency. "
    "Use the full 1-5 scale and the rubric anchors exactly."
)

SECTION_SUMMARY_PROMPT = """Summarise the following section of the IMF {doc} for {country}.
Constraints:
- At most {target} words (hard limit: fewer than {limit} words). One paragraph, no bullets, no headings.
- Lead with the section's main conclusion, then the 2-3 most decision-relevant facts (keep figures with their years and units) and any policy recommendation or risk.
- Use only information in the section text. If the text is mostly chart captions or table rows, summarise what they show.
- Do not begin with "This section" or repeat the title.

Section title: {title}
Section text:
<<<
{text}
>>>"""

REFINE_SUFFIX = """

An independent reviewer audited your previous draft.
Previous draft: {draft}
Reviewer feedback: {feedback}
Unsupported claims to remove: {unsupported}
Missing points to include if space allows: {missing}
Write an improved summary that fixes these issues and still obeys every constraint above."""

COMPRESS_PROMPT = """Rewrite the summary below in at most {target} words. Keep the key figures; add nothing new.
Summary: {text}"""

SUMMARY_JUDGE_PROMPT = """Evaluate the CANDIDATE summary against the SOURCE text.
Score each criterion with an integer from 1 to 5:
- consistency: 5 = no contradiction with the source and internally coherent; 3 = minor imprecision (e.g. rounded or mis-dated figure); 1 = contradicts the source.
- completeness: 5 = captures the main conclusion and the most decision-relevant facts; 3 = misses one key point; 1 = misses most key points.
- factual_alignment: 5 = every number, date and claim is traceable to the source; 3 = one unsupported or mis-attributed claim; 1 = several hallucinated claims.
List every unsupported claim (quote it) and up to 3 important missing points.
verdict = "pass" only if every score >= {threshold} and there are no unsupported claims, otherwise "revise".
Constraint context: {constraint_note}

SOURCE:
<<<
{source}
>>>

CANDIDATE:
<<<
{candidate}
>>>"""

EXEC_SUMMARY_PROMPT = """Write an executive summary of the COMPLETE IMF Article IV report on {country} for a financial-stability policymaker.
Return 6-10 bullets. Each bullet is ONE sentence (max 35 words) stating a primary financial-stability finding or risk assessment, with its key figure and year where the source gives one.
Cover, where the report addresses them: growth outlook; inflation; fiscal position and public debt; banking-sector soundness (capital, NPLs, provisioning); non-bank / credit-union risks; external position; main downside risks and the IMF's headline recommendation.
For each bullet also return: theme (one of: {themes}; or "Growth & Outlook"; or "Fiscal & External") and evidence_quote copied VERBATIM from the source (max 30 words).

SOURCE:
<<<
{text}
>>>"""



class SummaryJudgement(BaseModel):
    consistency: int = Field(description="1-5")
    completeness: int = Field(description="1-5")
    factual_alignment: int = Field(description="1-5")
    unsupported_claims: List[str] = Field(default_factory=list)
    missing_points: List[str] = Field(default_factory=list)
    verdict: str = Field(description="pass or revise")
    feedback: str = ""


class ExecBullet(BaseModel):
    point: str
    theme: str
    evidence_quote: str


class ExecutiveSummary(BaseModel):
    bullets: List[ExecBullet]


PROMPTS: Dict[str, str] = {"SYSTEM_ANALYST": SYSTEM_ANALYST, "SYSTEM_JUDGE": SYSTEM_JUDGE,
                           "SECTION_SUMMARY_PROMPT": SECTION_SUMMARY_PROMPT, "REFINE_SUFFIX": REFINE_SUFFIX,
                           "COMPRESS_PROMPT": COMPRESS_PROMPT, "SUMMARY_JUDGE_PROMPT": SUMMARY_JUDGE_PROMPT,
                           "EXEC_SUMMARY_PROMPT": EXEC_SUMMARY_PROMPT}

# %% [markdown]
# ### 1.6 Quote verification (deterministic hallucination check)
# A quote counts as **verified** when ≥ 80% of its word 4-grams occur in the cleaned source. This tolerates whitespace and punctuation differences from PDF extraction but rejects paraphrased or invented "quotes".

# %%
def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9.%]+", normalise_text(text).lower())


FULL_TEXT = "\n\n".join(s.prose_text + "\n" + "\n".join(l for _, l in s.tables) for s in SECTIONS)
_SOURCE_WORDS = _norm_words(FULL_TEXT)
SOURCE_4GRAMS = {tuple(_SOURCE_WORDS[i:i + 4]) for i in range(len(_SOURCE_WORDS) - 3)}


def quote_support(quote: str) -> float:
    w = _norm_words(quote or "")
    if len(w) < 4:
        return 1.0 if w and " ".join(w) in " ".join(_SOURCE_WORDS) else 0.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return sum(g in SOURCE_4GRAMS for g in grams) / len(grams)


def quote_verified(quote: str, threshold: float = 0.8) -> bool:
    return quote_support(quote) >= threshold


print("Sanity check – real sentence:", round(quote_support(" ".join(_SOURCE_WORDS[500:530])), 2),
      "| invented sentence:", round(quote_support("Dominica's central bank raised its policy rate to 7 percent in 2023"), 2))

# %% [markdown]
# ### 1.7 Section summaries with the generate → judge → refine loop
# For every major section:
# 1. The **extractor** writes a summary under 100 words. Sections above `section_max_tokens` are summarised map-reduce style.
# 2. A **code check** counts the words. A summary over the limit is compressed once and the violation is logged.
# 3. The **judge** (a different model) scores consistency, completeness and factual alignment against the section text.
# 4. If the verdict is `revise`, the judge's feedback is sent back to the extractor for one refinement round, and the new draft is **re-judged**. Both rounds are kept, so the report can show whether feedback actually helps.

# %%
def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'.%-]+\b", text))


def section_source(sec: Section, table_token_budget: int = 1500) -> str:
    """Prose plus a token-capped slice of table rows (for chart/table-heavy sections)."""
    src = sec.prose_text
    if sec.tables and sec.n_words < 400:
        rows, used = [], 0
        for _, line in sec.tables:
            used += count_tokens(line)
            if used > table_token_budget:
                break
            rows.append(line)
        src += "\n\n[Table extracts]\n" + "\n".join(rows)
    return src


def judge_summary(source: str, candidate: str, tag: str, constraint_note: Optional[str] = None) -> Optional[SummaryJudgement]:
    note = constraint_note or (f"the word limit is fewer than {CONFIG['summary_word_limit']} words and the "
                               f"candidate has {word_count(candidate)} words.")
    prompt = SUMMARY_JUDGE_PROMPT.format(threshold=CONFIG["judge_pass_threshold"], constraint_note=note,
                                         source=source, candidate=candidate)
    res = llm.generate(prompt, JUDGE_MODEL, system=SYSTEM_JUDGE, schema=SummaryJudgement, tag=tag)
    if not res.ok:
        print(f"  judge failed for {tag}: {res.error}")
        return None
    j = res.parsed
    for f_ in ("consistency", "completeness", "factual_alignment"):        # clamp to the rubric range
        setattr(j, f_, int(min(5, max(1, getattr(j, f_)))))
    return j


def judge_passes(j: Optional[SummaryJudgement]) -> bool:
    if j is None:
        return True                                        # nothing to act on – logged as missing score
    t = CONFIG["judge_pass_threshold"]
    return min(j.consistency, j.completeness, j.factual_alignment) >= t and not j.unsupported_claims


def generate_summary(title: str, text: str, doc: str, tag: str) -> Tuple[str, Dict[str, Any]]:
    """Single summarisation call (with map-reduce for very long inputs) + word-limit repair."""
    meta = {"map_reduce": False, "compressed": False}
    if count_tokens(text) > CONFIG["section_max_tokens"]:
        meta["map_reduce"] = True
        paras, parts, buf = text.split("\n\n"), [], ""
        for p in paras:
            if buf and count_tokens(buf + p) > CONFIG["section_max_tokens"] // 2:
                parts.append(buf)
                buf = ""
            buf += p + "\n\n"
        parts.append(buf)
        partials = []
        for k, part in enumerate(parts):
            r = llm.generate(SECTION_SUMMARY_PROMPT.format(doc=doc, country=C, target=120, limit=150,
                                                           title=f"{title} (part {k + 1})", text=part),
                             EXTRACTOR_MODEL, system=SYSTEM_ANALYST, tag=f"{tag}-map{k}")
            partials.append(r.text.strip())
        text = "\n\n".join(partials)
    res = llm.generate(SECTION_SUMMARY_PROMPT.format(doc=doc, country=C, target=CONFIG["summary_target_words"],
                                                     limit=CONFIG["summary_word_limit"], title=title, text=text),
                       EXTRACTOR_MODEL, system=SYSTEM_ANALYST, tag=tag)
    summary = res.text.strip()
    meta.update({"latency_s": res.latency_s, "error": res.error, "words_initial": word_count(summary)})
    if word_count(summary) >= CONFIG["summary_word_limit"]:
        meta["compressed"] = True
        summary = llm.generate(COMPRESS_PROMPT.format(target=CONFIG["summary_target_words"] - 10, text=summary),
                               EXTRACTOR_MODEL, system=SYSTEM_ANALYST, tag=f"{tag}-compress").text.strip()
    return summary, meta


def summarise_with_judge(title: str, source: str, doc: str, tag: str) -> Dict[str, Any]:
    summary, meta = generate_summary(title, source, doc, tag)
    j = judge_summary(source, summary, f"{tag}-judge0")
    rounds = [{"round": 0, "summary": summary, "words": word_count(summary),
               "judge": j.model_dump() if j else None}]
    for r in range(1, CONFIG["max_refinement_rounds"] + 1):
        if judge_passes(j):
            break
        prompt = SECTION_SUMMARY_PROMPT.format(doc=doc, country=C, target=CONFIG["summary_target_words"],
                                               limit=CONFIG["summary_word_limit"], title=title, text=source)
        prompt += REFINE_SUFFIX.format(draft=summary, feedback=j.feedback, unsupported=j.unsupported_claims or "none",
                                       missing=j.missing_points or "none")
        summary = llm.generate(prompt, EXTRACTOR_MODEL, system=SYSTEM_ANALYST, tag=f"{tag}-refine{r}").text.strip()
        if word_count(summary) >= CONFIG["summary_word_limit"]:
            summary = llm.generate(COMPRESS_PROMPT.format(target=CONFIG["summary_target_words"] - 10, text=summary),
                                   EXTRACTOR_MODEL, system=SYSTEM_ANALYST, tag=f"{tag}-refine{r}-compress").text.strip()
        j = judge_summary(source, summary, f"{tag}-judge{r}")
        rounds.append({"round": r, "summary": summary, "words": word_count(summary),
                       "judge": j.model_dump() if j else None})
    return {"final_summary": summary, "final_words": word_count(summary),
            "within_limit": word_count(summary) < CONFIG["summary_word_limit"], "rounds": rounds, **meta}


SUMMARISABLE = [s for s in SECTIONS if s.part != "Cover" and (s.n_words >= 80 or s.tables)]
SECTION_SUMMARIES: List[Dict[str, Any]] = []
for sec in SUMMARISABLE:
    doc = sec.part if sec.part in WHOLE_PARTS else "Staff Report"
    out = summarise_with_judge(sec.title, section_source(sec), doc, tag=f"summary-{sec.sid}")
    SECTION_SUMMARIES.append({"sid": sec.sid, "part": sec.part, "title": sec.title,
                              "pages": f"{sec.start_page}-{sec.end_page}", "source_words": sec.n_words, **out})
    print(f"{sec.sid} {sec.title[:45]:<45} words={out['final_words']:>3} rounds={len(out['rounds'])}")
save_json(SECTION_SUMMARIES, "summaries.json")


def judge_rows(items: List[Dict[str, Any]], kind: str) -> pd.DataFrame:
    rows = []
    for it in items:
        for rd in it["rounds"]:
            if rd["judge"]:
                rows.append({"kind": kind, "id": it["sid"], "title": it["title"], "round": rd["round"],
                             "words": rd["words"], **{k: rd["judge"][k] for k in
                                                     ("consistency", "completeness", "factual_alignment")},
                             "n_unsupported": len(rd["judge"]["unsupported_claims"]),
                             "verdict": rd["judge"]["verdict"]})
    return pd.DataFrame(rows)


SUMMARY_JUDGE_DF = judge_rows(SECTION_SUMMARIES, "section_summary")
display(pd.DataFrame([{"Section": s["title"], "Pages": s["pages"], "Words": s["final_words"],
                       "Summary": s["final_summary"]} for s in SECTION_SUMMARIES]))
if not SUMMARY_JUDGE_DF.empty:
    display(SUMMARY_JUDGE_DF.groupby("round")[["consistency", "completeness", "factual_alignment",
                                                "n_unsupported"]].mean().round(2))
print("Word-limit compliance:", sum(s["within_limit"] for s in SECTION_SUMMARIES), "/", len(SECTION_SUMMARIES),
      "| needed compression:", sum(s["compressed"] for s in SECTION_SUMMARIES))

# %% [markdown]
# ### 1.8 Executive summary of the full report
# The narrative of the whole package (press release, staff report and annexes, ED statement, ≈ 40–60k tokens) fits comfortably in Gemini's context window. The executive summary is therefore generated from the **full source text**, not by summarising summaries. That avoids the information loss of map-reduce. If the text ever exceeded `exec_summary_max_input_tokens`, the code would fall back to the section summaries. Each bullet's evidence quote is checked deterministically, and the judge audits the whole summary.

# %%
EXEC_SOURCE = "\n\n".join(f"## {s.title} ({s.part}, pp.{s.start_page}-{s.end_page})\n{s.prose_text}"
                          for s in SECTIONS if s.part != "Cover" and s.n_words > 0)
exec_input_tokens = count_tokens(EXEC_SOURCE)
if exec_input_tokens > CONFIG["exec_summary_max_input_tokens"]:
    EXEC_SOURCE = "\n\n".join(f"## {s['title']}\n{s['final_summary']}" for s in SECTION_SUMMARIES)
    exec_mode = "section-summaries (map-reduce fallback)"
else:
    exec_mode = "full narrative text"
print(f"Executive-summary input: {exec_mode}, ~{count_tokens(EXEC_SOURCE):,} tokens")

exec_res = llm.generate(EXEC_SUMMARY_PROMPT.format(country=C, themes="; ".join(THEMES), text=EXEC_SOURCE),
                        EXTRACTOR_MODEL, system=SYSTEM_ANALYST, schema=ExecutiveSummary, tag="exec-summary")
EXEC_BULLETS = exec_res.parsed.bullets if exec_res.ok else []
if not exec_res.ok:
    print("Executive summary failed:", exec_res.error)
exec_text = "\n".join(f"- {b.point}" for b in EXEC_BULLETS)
exec_judge = judge_summary(EXEC_SOURCE, exec_text, "exec-judge",
                           constraint_note=f"6-10 one-sentence bullets of <= 35 words; the candidate has {len(EXEC_BULLETS)} bullets."
                           ) if EXEC_BULLETS else None
EXEC_SUMMARY = {
    "mode": exec_mode, "model": EXTRACTOR_MODEL, "latency_s": exec_res.latency_s,
    "n_bullets": len(EXEC_BULLETS), "bullet_count_ok": 6 <= len(EXEC_BULLETS) <= 10,
    "bullets": [{**b.model_dump(), "words": word_count(b.point),
                 "quote_support": round(quote_support(b.evidence_quote), 3),
                 "quote_verified": quote_verified(b.evidence_quote)} for b in EXEC_BULLETS],
    "judge": exec_judge.model_dump() if exec_judge else None,
}
save_json(EXEC_SUMMARY, "executive_summary.json")
display(Markdown("**Executive summary**\n\n" + exec_text))
print("Quotes verified:", sum(b["quote_verified"] for b in EXEC_SUMMARY["bullets"]), "/", len(EXEC_BULLETS))
if exec_judge:
    print("Judge:", {k: getattr(exec_judge, k) for k in ("consistency", "completeness", "factual_alignment", "verdict")})
