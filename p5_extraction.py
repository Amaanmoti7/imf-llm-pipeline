# %% [markdown]
# ---
# ## Task 2 — Thematic extraction, indicators and structured output
# ### 2.1 Thematic distribution across the 7 financial-stability themes
# Two complementary measures are computed:
# 1. **Lexicon mention counts.** Each theme has a curated set of regex patterns (multi-word terms included). Counts are reported raw and **per 1,000 words** so the result doesn't depend on document length. The lexicon is transparent and reproducible, but it can over-count generic words like *bank* or *credit*.
# 2. **Embedding-based chunk classification.** Each prose chunk goes to the theme whose description is nearest in embedding space. The bottom decile of best-match similarity is labelled "Other". This captures what a chunk is *about*, not which words it repeats.
#
# Where the two measures disagree is itself worth discussing in the report: it exposes lexicon bias.

# %%
THEME_LEXICON: Dict[str, List[str]] = {
    "Inflation & Prices": [r"inflation\w*", r"disinflation", r"consumer prices?", r"\bCPI\b", r"cost[- ]of[- ]living",
                           r"price (?:level|pressures?|increases?|shocks?|controls?|stability)",
                           r"(?:food|fuel|energy|commodity|oil|import) prices?", r"deflat\w*"],
    "Interest Rates & Credit": [r"interest rates?", r"policy rates?", r"discount rate", r"lending rates?",
                                r"deposit rates?", r"\bcredit\b(?!\s+unions?)", r"lending", r"\bloans?\b",
                                r"borrowing costs?", r"spreads?", r"financial intermediation", r"access to finance"],
    "Housing & Real Estate": [r"housing", r"real estate", r"mortgages?", r"propert(?:y|ies)", r"residential",
                              r"construction", r"house prices?", r"dwellings?", r"foreclosures?", r"\bland\b"],
    "Banking Capital & Liquidity": [r"capital adequacy", r"regulatory capital", r"capital (?:ratios?|buffers?|requirements?)",
                                    r"\btier[- ]?1\b", r"liquidity", r"liquid assets", r"non-?performing loans?",
                                    r"\bNPLs?\b", r"provision\w*", r"credit unions?", r"\bbanks?\b", r"banking",
                                    r"deposits?", r"profitability", r"return on (?:assets|equity)"],
    "Sovereign Debt": [r"public debt", r"government debt", r"sovereign", r"debt[- ]to[- ]GDP", r"debt sustainability",
                       r"\bDSA\b", r"debt distress", r"debt service", r"fiscal (?:rule|anchor|space|consolidation|buffers?)",
                       r"external debt", r"guaranteed debt", r"arrears", r"gross financing needs?"],
    "Labor Market Dynamics": [r"labou?r (?:market|force|mobility|productivity|participation)", r"unemploy\w*",
                              r"\bemployment\b", r"\bwages?\b", r"\bworkers?\b", r"\bjobs?\b", r"\bskills?\b",
                              r"\byouth\b", r"emigration", r"pensions?", r"social security"],
    "Systemic Financial Risk": [r"systemic", r"financial stability", r"financial (?:sector )?risks?", r"contagion",
                                r"spillovers?", r"stress tests?", r"macroprudential", r"interconnect\w*",
                                r"de-?risking", r"correspondent bank\w*", r"AML/CFT", r"money laundering",
                                r"\bcrisis\b", r"supervis\w*", r"resolution framework"],
}
THEME_DESCRIPTIONS = {
    "Inflation & Prices": "inflation, consumer prices, cost of living, food and fuel price pressures",
    "Interest Rates & Credit": "interest rates, bank lending and private sector credit growth, borrowing costs, access to finance",
    "Housing & Real Estate": "housing, real estate, mortgages, property markets and construction",
    "Banking Capital & Liquidity": "bank and credit union capital adequacy, liquidity, nonperforming loans and provisioning",
    "Sovereign Debt": "public debt, fiscal deficits, debt sustainability analysis, government financing and the fiscal rule",
    "Labor Market Dynamics": "employment, unemployment, wages, labor force participation, skills, pensions and social security",
    "Systemic Financial Risk": "systemic financial stability risks, contagion, supervision, AML/CFT, correspondent banking, stress tests",
}


def lexicon_counts(text: str) -> Dict[str, int]:
    out = {}
    for theme, pats in THEME_LEXICON.items():
        flags = 0
        out[theme] = sum(len(re.findall(p, text, flags=flags if re.search(r"[A-Z]{2,}", p) else re.IGNORECASE))
                         for p in pats)
    return out


PROSE_TEXT = "\n".join(s.prose_text for s in SECTIONS if s.part != "Cover")
n_words_total = len(PROSE_TEXT.split())
lex = lexicon_counts(PROSE_TEXT)

theme_emb = VINDEX.model.encode([THEME_DESCRIPTIONS[t] for t in THEMES], normalize_embeddings=True)
prose_idx = [i for i, c in enumerate(CHUNKS) if c.kind == "prose"]
sims = VINDEX.emb[prose_idx] @ theme_emb.T
# Forced-choice assignment; chunks in the bottom decile of best-match similarity are labelled "Other"
# (weakly thematic text such as mission logistics or energy-sector detail). A relative floor is used because
# absolute cosine levels differ across embedding models.
SIM_FLOOR = float(np.quantile(sims.max(axis=1), 0.10)) if len(prose_idx) else 0.0
chunk_theme = [THEMES[j] if sims[r, j] >= SIM_FLOOR else "Other" for r, j in enumerate(sims.argmax(axis=1))]
emb_counts = Counter(chunk_theme)

THEME_DF = pd.DataFrame({
    "theme": THEMES,
    "lexicon_mentions": [lex[t] for t in THEMES],
    "mentions_per_1k_words": [round(1000 * lex[t] / n_words_total, 2) for t in THEMES],
    "lexicon_share_pct": [round(100 * lex[t] / max(1, sum(lex.values())), 1) for t in THEMES],
    "embedding_chunks": [emb_counts.get(t, 0) for t in THEMES],
    "embedding_share_pct": [round(100 * emb_counts.get(t, 0) / max(1, len(prose_idx)), 1) for t in THEMES],
}).sort_values("lexicon_mentions", ascending=False)

per_part = pd.DataFrame({p: lexicon_counts("\n".join(s.prose_text for s in SECTIONS if s.part == p))
                         for p in PART_ORDER[1:] if any(s.part == p for s in SECTIONS)}).T
display(THEME_DF)
display(per_part)
print(f"Prose chunks labelled Other (best similarity < {SIM_FLOOR:.3f}, bottom decile): {emb_counts.get('Other', 0)}")
save_json({"method": {"lexicon": THEME_LEXICON, "embedding_descriptions": THEME_DESCRIPTIONS,
                      "similarity_floor": SIM_FLOOR, "prose_words": n_words_total},
           "distribution": THEME_DF.to_dict(orient="records"),
           "by_document_part": per_part.reset_index(names="part").to_dict(orient="records"),
           "chunk_assignments": [{"chunk_id": CHUNKS[i].chunk_id, "theme": t, "similarity": round(float(sims[r].max()), 3)}
                                 for r, (i, t) in enumerate(zip(prose_idx, chunk_theme))]}, "themes.json")

# %% [markdown]
# ### 2.2 Qualitative risk insights: growth drivers and vulnerabilities
# The extractor reads the full narrative and returns 5–7 **growth drivers / strengths** and 5–7 **systemic vulnerabilities / challenges**. Each one comes with a verbatim evidence quote. The pipeline then checks the item counts, verifies every quote against the source, and has the judge audit the list.

# %%
INSIGHTS_PROMPT = """From the IMF Article IV report on {country} below, extract:
1. growth_drivers: 5-7 national macroeconomic growth drivers or strengths (e.g. resilient sectors, policy buffers, sound institutions).
2. vulnerabilities: 5-7 systemic vulnerabilities or challenges (e.g. fiscal, financial-sector, external, climate risks).
For each item: title (max 8 words), explanation (max 35 words, include a figure if the report gives one),
theme (one of: {themes}; or "Growth & Outlook"; or "Fiscal & External"; or "Climate & Structural"),
and evidence_quote copied VERBATIM from the source (max 30 words).
Items must be distinct and specific to {country}; do not list generic risks the report does not discuss.

SOURCE:
<<<
{text}
>>>"""
PROMPTS["INSIGHTS_PROMPT"] = INSIGHTS_PROMPT


class Insight(BaseModel):
    title: str
    explanation: str
    theme: str
    evidence_quote: str


class RiskInsights(BaseModel):
    growth_drivers: List[Insight]
    vulnerabilities: List[Insight]


ins_res = llm.generate(INSIGHTS_PROMPT.format(country=C, themes="; ".join(THEMES), text=EXEC_SOURCE),
                       EXTRACTOR_MODEL, system=SYSTEM_ANALYST, schema=RiskInsights, tag="risk-insights")
if ins_res.ok:
    INSIGHTS = ins_res.parsed
else:
    print("Insight extraction failed:", ins_res.error)
    INSIGHTS = RiskInsights(growth_drivers=[], vulnerabilities=[])


def insights_as_text(ins: RiskInsights) -> str:
    return "\n".join([f"STRENGTH: {i.title} - {i.explanation}" for i in ins.growth_drivers] +
                     [f"VULNERABILITY: {i.title} - {i.explanation}" for i in ins.vulnerabilities])


ins_judge = judge_summary(EXEC_SOURCE, insights_as_text(INSIGHTS), "insights-judge",
                          constraint_note="5-7 strengths and 5-7 vulnerabilities, each specific and supported by the source.")
RISK_INSIGHTS = {
    "model": EXTRACTOR_MODEL,
    "count_ok": {"growth_drivers": 5 <= len(INSIGHTS.growth_drivers) <= 7,
                 "vulnerabilities": 5 <= len(INSIGHTS.vulnerabilities) <= 7},
    **{k: [{**i.model_dump(), "quote_support": round(quote_support(i.evidence_quote), 3),
            "quote_verified": quote_verified(i.evidence_quote)} for i in getattr(INSIGHTS, k)]
       for k in ("growth_drivers", "vulnerabilities")},
    "judge": ins_judge.model_dump() if ins_judge else None,
}
save_json(RISK_INSIGHTS, "risk_insights.json")
for k in ("growth_drivers", "vulnerabilities"):
    display(pd.DataFrame(RISK_INSIGHTS[k])[["title", "theme", "explanation", "quote_verified"]]
            .rename(columns={"title": k.replace("_", " ").title()}))
if ins_judge:
    print("Judge:", {k: getattr(ins_judge, k) for k in ("consistency", "completeness", "factual_alignment", "verdict")})

# %% [markdown]
# ### 2.3 Core indicator JSON: RAG-grounded, schema-constrained extraction
# **Approach.** Each indicator has a precise **definition**, plus two retrieval queries (one targeting prose, one targeting tables). Their top-ranked chunks are merged into a single numbered context. The extractor then fills a Pydantic schema enforced through Gemini's `response_schema`. For every field it returns the value, reference year, status, a verbatim quote and the source chunk id. A separate **proxy** slot keeps a "close but not equal" measure (total regulatory capital vs Tier 1, say) out of the target field.
#
# **Fallbacks.** Missing values become JSON `null`, not `-1.0`. A `-1.0` sentinel is a valid-looking number: it would pass type checks and then plot as a real negative rate. Each `null` carries a machine-readable reason (`not_reported`, `ambiguous`, `failed_validation`), which the report explains in domain terms.
#
# **Sanity rules.** Plausible-range checks catch scale errors, such as 0.046 written instead of 4.6%.

# %%
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {
    "gdp_growth_forecast_pct": {
        "definition": "Real GDP growth (annual % change) PROJECTED by IMF staff for the first forecast year (the year the report is published). Not a historical outturn.",
        "queries": ["real GDP growth projected outlook for this year", "Selected Economic Indicators real GDP annual percent change projected"],
        "range": (-30, 30)},
    "headline_inflation_rate_pct": {
        "definition": "Headline consumer price inflation, PERIOD-AVERAGE annual % change, for the latest actual or estimated year (not a projection). If only end-of-period is stated, use it and say so in note.",
        "queries": ["consumer prices inflation period average annual percent change", "inflation declined from its peak to percent"],
        "range": (-10, 60)},
    "central_bank_policy_rate_pct": {
        "definition": "The monetary-policy (policy/discount) rate set by the central bank responsible for the country; for a currency-union member, the common central bank's rate. Only if a numeric rate is explicitly stated.",
        "queries": ["central bank monetary policy rate interest rate", "currency union common central bank monetary policy and exchange rate arrangement"],
        "range": (-1, 40)},
    "non_performing_loan_ratio": {
        "definition": "Nonperforming loans to total gross loans (%) for COMMERCIAL BANKS, latest year. Do NOT use credit-union or system-wide figures for this field.",
        "queries": ["commercial banks nonperforming loans to total gross loans", "bank NPL ratio provisioning asset quality"],
        "range": (0, 100)},
    "unemployment_rate_pct": {
        "definition": "Unemployment rate (% of labour force), the latest value stated anywhere in the report; give its reference year.",
        "queries": ["unemployment rate labor market", "social indicators population unemployment rate literacy"],
        "range": (0, 60)},
    "tier_1_capital_ratio": {
        "definition": "Tier 1 capital to risk-weighted assets (%) for banks. If only a different capital measure is reported (e.g. total regulatory capital to risk-weighted assets), value must be null and that figure goes in proxy_value with proxy_definition.",
        "queries": ["tier 1 capital to risk-weighted assets banks", "regulatory capital ratio capital adequacy commercial banks"],
        "range": (0, 100)},
}
INDICATORS = list(INDICATOR_SPECS)


class IndicatorValue(BaseModel):
    value: Optional[float] = Field(None, description="number in percent units, or null if not reported")
    reference_year: Optional[str] = None
    status: str = Field(description="extracted | not_reported | ambiguous")
    source_quote: Optional[str] = Field(None, description="verbatim text (for tables: title + row label + value)")
    source_chunk: Optional[str] = Field(None, description="chunk id such as C012")
    proxy_value: Optional[float] = None
    proxy_definition: Optional[str] = None
    note: str = ""


class IndicatorExtraction(BaseModel):
    country_name: str
    gdp_growth_forecast_pct: IndicatorValue
    headline_inflation_rate_pct: IndicatorValue
    central_bank_policy_rate_pct: IndicatorValue
    non_performing_loan_ratio: IndicatorValue
    unemployment_rate_pct: IndicatorValue
    tier_1_capital_ratio: IndicatorValue


class CoreIndicators(BaseModel):
    """The flat target schema required by the brief."""
    country_name: str
    gdp_growth_forecast_pct: Optional[float]
    headline_inflation_rate_pct: Optional[float]
    central_bank_policy_rate_pct: Optional[float]
    non_performing_loan_ratio: Optional[float]
    unemployment_rate_pct: Optional[float]
    tier_1_capital_ratio: Optional[float]


INDICATOR_SYSTEM = (
    f"You are a precise data-extraction engine for IMF country reports on {C}. Rules: "
    "(1) Extract a value ONLY if it is explicitly stated in the CONTEXT for this country. "
    "(2) Never compute, convert, infer or use outside knowledge (even if you know the value). "
    "(3) If an indicator is not reported, set value=null, status='not_reported' and explain why in note. "
    "(4) Percentages are written in percent units (4.6 means 4.6%), never as fractions. "
    "(5) source_quote must be copied verbatim; for tables give the table title, the row label and the value. "
    "(6) Tables may list several institution types (e.g. commercial banks vs credit unions): use the row under the "
    "correct table title. "
    f"(7) Do not confuse {C} with {CONFIG['confusable_country']}."
)
INDICATOR_PROMPT = """Extract the core indicators for {country} from the CONTEXT passages of {report}.

Indicator definitions:
{definitions}

Return every field. Cite the chunk id (e.g. C012) that contains each value.

CONTEXT:
{context}"""
GENERIC_INDICATOR_PROMPT = """Here is text from an IMF report about {country}. Extract these indicators and return them as JSON with the keys country_name, gdp_growth_forecast_pct, headline_inflation_rate_pct, central_bank_policy_rate_pct, non_performing_loan_ratio, unemployment_rate_pct, tier_1_capital_ratio.

{context}"""
PROMPTS.update({"INDICATOR_SYSTEM": INDICATOR_SYSTEM, "INDICATOR_PROMPT": INDICATOR_PROMPT,
                "GENERIC_INDICATOR_PROMPT": GENERIC_INDICATOR_PROMPT})


def build_indicator_context() -> Tuple[str, List[str]]:
    hits = []
    for spec in INDICATOR_SPECS.values():
        for q in spec["queries"]:
            hits += VINDEX.search(q, k=3, kind="prose") + VINDEX.search(q, k=3, kind="table")
    ctx = format_context(hits, max_tokens=16_000)
    return ctx, sorted({c.chunk_id for c, _ in hits})


INDICATOR_CONTEXT, INDICATOR_CHUNKS = build_indicator_context()
DEFINITIONS = "\n".join(f"- {k}: {v['definition']}" for k, v in INDICATOR_SPECS.items())
print(f"Indicator context: {len(INDICATOR_CHUNKS)} chunks, ~{count_tokens(INDICATOR_CONTEXT):,} tokens")


def flatten_generic(raw: Any) -> Dict[str, Any]:
    """Map a free-form JSON reply onto the flat schema (accepts {'value': x} objects too)."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k in ["country_name"] + INDICATORS:
        v = raw.get(k)
        if isinstance(v, dict):
            v = v.get("value")
        out[k] = v
    return out


def schema_compliance(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Strict check: all keys present, numeric fields are float/int/null (no '4.6%' strings)."""
    missing = [k for k in ["country_name"] + INDICATORS if k not in flat]
    bad_types = [k for k in INDICATORS if k in flat and flat[k] is not None and not isinstance(flat[k], (int, float))]
    return {"valid_json": bool(flat), "missing_keys": missing, "bad_types": bad_types,
            "compliant": bool(flat) and not missing and not bad_types}


def run_indicator_extraction(model: str, style: str) -> Dict[str, Any]:
    """style = 'strict' (system instruction + response_schema) or 'generic' (plain user prompt, no schema)."""
    if style == "strict":
        res = llm.generate(INDICATOR_PROMPT.format(country=C, report=CONFIG["imf_country_report_no"],
                                                   definitions=DEFINITIONS, context=INDICATOR_CONTEXT),
                           model, system=INDICATOR_SYSTEM, schema=IndicatorExtraction, tag=f"indicators-{style}-{model}")
        detail = res.parsed.model_dump() if res.ok else {}
        flat = {"country_name": detail.get("country_name")} if detail else {}
        flat.update({k: detail[k]["value"] for k in INDICATORS if detail})
    else:
        res = llm.generate(GENERIC_INDICATOR_PROMPT.format(country=C, context=INDICATOR_CONTEXT),
                           model, tag=f"indicators-{style}-{model}")
        detail, flat = {}, flatten_generic(extract_json_block(res.text))
    return {"model": model, "style": style, "latency_s": round(res.latency_s, 2), "cached": res.cached,
            "error": res.error, "flat": flat, "detail": detail, "compliance": schema_compliance(flat)}


def apply_fallbacks(run: Dict[str, Any]) -> Tuple[CoreIndicators, Dict[str, Dict[str, Any]]]:
    """Coerce to the target schema; every null gets an explicit, machine-readable fallback reason."""
    flat, detail, reasons = dict(run["flat"]), run["detail"], {}
    for k in INDICATORS:
        v = flat.get(k)
        if isinstance(v, str):                                # e.g. "4.6%" from generic prompts
            m = re.search(r"-?\d+(?:\.\d+)?", v)
            v = float(m.group(0)) if m else None
            reasons[k] = {"fallback": "coerced_from_string"} if v is not None else {"fallback": "unparseable_string"}
        lo, hi = INDICATOR_SPECS[k]["range"]
        if v is not None and not (lo <= v <= hi):
            reasons[k] = {"fallback": "out_of_plausible_range", "raw": v}
            v = None
        if v is None and k not in reasons:
            d = detail.get(k, {}) if detail else {}
            reasons[k] = {"fallback": d.get("status") or ("extraction_failed" if run["error"] else "not_reported"),
                          "model_note": d.get("note"), "proxy_value": d.get("proxy_value"),
                          "proxy_definition": d.get("proxy_definition")}
        flat[k] = v
    flat["country_name"] = flat.get("country_name") or C
    return CoreIndicators(**flat), reasons


MAIN_RUN = run_indicator_extraction(EXTRACTOR_MODEL, "strict")
CORE, FALLBACKS = apply_fallbacks(MAIN_RUN)
print("Schema compliance:", MAIN_RUN["compliance"])
display(pd.DataFrame([{"indicator": k, "value": getattr(CORE, k),
                       "year": MAIN_RUN["detail"].get(k, {}).get("reference_year") if MAIN_RUN["detail"] else None,
                       "status": MAIN_RUN["detail"].get(k, {}).get("status") if MAIN_RUN["detail"] else None,
                       "source": MAIN_RUN["detail"].get(k, {}).get("source_chunk") if MAIN_RUN["detail"] else None,
                       "quote": (MAIN_RUN["detail"].get(k, {}).get("source_quote") or "")[:90] if MAIN_RUN["detail"] else "",
                       "fallback": FALLBACKS.get(k, {}).get("fallback")} for k in INDICATORS]))

# %% [markdown]
# ### 2.4 Evaluator validation of the JSON: judge audit + human ground truth
# Two independent checks:
# * **LLM-as-a-judge audit.** The judge sees each extracted value, its quote and the *same* retrieved context, and labels it `supported`, `unsupported`, `contradicted`, `correct_null` or `missed_value`.
# * **Human ground truth** (`ground_truth.json`). Values I verified by hand against the PDF pages, with accepted alternatives where the definition is genuinely ambiguous (end-of-period vs period-average inflation, for example). This measures **accuracy**, and also measures how far the judge itself can be trusted: where the judge and the ground truth disagree, the judge's error rate shows.

# %%
INDICATOR_AUDIT_PROMPT = """Audit these indicator values extracted for {country} against the CONTEXT.
For each indicator choose a verdict:
- supported: the value matches the context and the indicator definition;
- unsupported: the value cannot be found in the context;
- contradicted: the context gives a different value or the value belongs to another series/year/institution type;
- correct_null: the value is null and the context indeed does not report this indicator;
- missed_value: the value is null but the context does report it (give expected_value).
Use the definitions strictly.

Definitions:
{definitions}

EXTRACTED (JSON):
{extracted}

CONTEXT:
{context}"""
PROMPTS["INDICATOR_AUDIT_PROMPT"] = INDICATOR_AUDIT_PROMPT


class IndicatorVerdict(BaseModel):
    indicator: str
    verdict: str
    expected_value: Optional[float] = None
    comment: str = ""


class IndicatorAudit(BaseModel):
    verdicts: List[IndicatorVerdict]
    overall_score: int = Field(description="1-5 overall factual accuracy")


def audit_indicators(run: Dict[str, Any], tag: str) -> Dict[str, Any]:
    extracted = run["detail"] or {k: {"value": run["flat"].get(k)} for k in INDICATORS}
    extracted = {k: {f: v for f, v in extracted.get(k, {}).items() if f in ("value", "reference_year", "source_quote",
                                                                          "source_chunk", "proxy_value")}
                 for k in INDICATORS}
    res = llm.generate(INDICATOR_AUDIT_PROMPT.format(country=C, definitions=DEFINITIONS,
                                                     extracted=json.dumps(extracted, indent=1), context=INDICATOR_CONTEXT),
                       JUDGE_MODEL, system=SYSTEM_JUDGE, schema=IndicatorAudit, tag=tag)
    if not res.ok:
        return {"error": res.error, "verdicts": {}, "support_rate": np.nan, "overall_score": np.nan}
    verdicts = {v.indicator: v.model_dump() for v in res.parsed.verdicts}
    good = sum(v["verdict"] in ("supported", "correct_null") for v in verdicts.values())
    return {"verdicts": verdicts, "support_rate": good / len(INDICATORS),
            "overall_score": int(min(5, max(1, res.parsed.overall_score)))}


GT_PATH = BASE_DIR / "ground_truth.json"
GROUND_TRUTH = json.loads(GT_PATH.read_text())["indicators"] if GT_PATH.exists() else {}
if not GROUND_TRUTH:
    print("⚠️  ground_truth.json not found – accuracy metrics will be NaN.")


def accuracy_vs_truth(flat: Dict[str, Any], tol: float = 0.051) -> Dict[str, Any]:
    per = {}
    for k in INDICATORS:
        gt = GROUND_TRUTH.get(k)
        if gt is None:
            continue
        accepted = [gt["value"]] + gt.get("accepted_alternatives", [])
        v = flat.get(k)
        if gt["value"] is None:
            per[k] = v is None
        else:
            per[k] = isinstance(v, (int, float)) and any(a is not None and abs(v - a) <= tol for a in accepted)
    return {"per_indicator": per, "accuracy": (sum(per.values()) / len(per)) if per else np.nan}


MAIN_AUDIT = audit_indicators(MAIN_RUN, "indicator-audit-main")
MAIN_ACC = accuracy_vs_truth(CORE.model_dump())
audit_table = pd.DataFrame([{
    "indicator": k, "extracted": getattr(CORE, k),
    "ground_truth": GROUND_TRUTH.get(k, {}).get("value"), "correct": MAIN_ACC["per_indicator"].get(k),
    "judge_verdict": MAIN_AUDIT["verdicts"].get(k, {}).get("verdict"),
    "judge_agrees_with_truth": (MAIN_AUDIT["verdicts"].get(k, {}).get("verdict") in ("supported", "correct_null"))
                               == MAIN_ACC["per_indicator"].get(k) if k in MAIN_ACC["per_indicator"] else None,
    "fallback": FALLBACKS.get(k, {}).get("fallback")} for k in INDICATORS])
display(audit_table)
print(f"Accuracy vs human ground truth: {MAIN_ACC['accuracy']:.0%} | judge support rate: {MAIN_AUDIT['support_rate']:.0%}")

save_json({"country": C, "report": CONFIG["imf_country_report_no"], "model": EXTRACTOR_MODEL,
           "indicators": CORE.model_dump(), "fallback_value": None, "fallbacks": FALLBACKS,
           "evidence": MAIN_RUN["detail"], "schema_compliance": MAIN_RUN["compliance"],
           "context_chunks": INDICATOR_CHUNKS}, "core_indicators.json")
save_json({"judge_model": JUDGE_MODEL, "audit": MAIN_AUDIT, "accuracy_vs_ground_truth": MAIN_ACC,
           "table": audit_table.to_dict(orient="records")}, "indicator_audit.json")

# %% [markdown]
# ### 2.5 Temporal trends for longitudinal plotting
# * **Deterministic table parser** (the reference). It walks the layout-mode lines, tracks the current table title, year header and row group (e.g. *Consumer prices → Period average*), and keeps a row only when its count of numbers **exactly equals** the count of year columns. Ragged rows are skipped and counted, never guessed. Years ≥ `projection_start_year` are flagged as projections.
# * **LLM trend identification.** The extractor scans the retrieved table chunks and proposes the series most relevant to financial stability. Every LLM value is then **cross-checked against the parser**, which gives an agreement rate. The dashboard plots the parser's values, so every number on the chart traces to a table cell.

# %%
def parse_number(tok: str) -> float:
    if MISSING_TOKEN.match(tok):
        return np.nan
    neg = tok.startswith("(") and tok.endswith(")")
    v = float(tok.strip("()%").replace(",", ""))
    return -v if neg else v


def split_row(line: str) -> Tuple[str, List[float]]:
    toks = line.split()
    nums = []
    while toks and (NUM_TOKEN.match(toks[-1]) or MISSING_TOKEN.match(toks[-1])):
        nums.insert(0, parse_number(toks.pop()))
    label = re.sub(r"\s*\b\d{1,2}/", "", " ".join(toks)).strip(" :")
    return label, nums


def parse_tables(pages: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, int]:
    rows, skipped = [], 0
    title, years, group = None, None, None
    for pg in pages:
        if not page_is_tabular(pg["lines"]) and not any(TITLE_LINE.match(l) for l in pg["lines"]):
            title, years, group = None, None, None          # a prose page ends any open table
        for raw in pg["layout_lines"]:
            s = re.sub(r"\s+", " ", raw).strip()
            if TITLE_LINE.match(s):
                title, years, group = (s if re.search(r"\bTable\b", s) else None), None, None
                continue
            if title is None:
                continue
            ys = years_in(s)
            numeric = [t for t in s.split() if NUM_TOKEN.match(t)]
            if len(ys) >= 4 and len(ys) >= 0.8 * len(numeric):
                years = ys
                continue
            label, nums = split_row(s)
            if not nums:
                if label and len(label) < 90 and not label.lower().startswith(("source", "note")):
                    group = label
                continue
            if years is None or not label or len(nums) < 3:
                continue                                    # a trailing year/number in prose is not a row
            if len(nums) != len(years):
                skipped += 1
                continue
            for y, v in zip(years, nums):
                if not np.isnan(v):
                    rows.append({"table": title, "group": group, "series": label, "year": y, "value": v,
                                 "page": pg["page"], "is_projection": y >= CONFIG["projection_start_year"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["table_key"] = df["table"].str.replace(r"\s*\(continued\)|\s*\(concluded\)", "", regex=True)
        df = df.drop_duplicates(subset=["table_key", "group", "series", "year"], keep="first")
    return df, skipped


TRENDS_DF, n_skipped = parse_tables(PAGES)
print(f"Parsed {len(TRENDS_DF):,} table cells from {TRENDS_DF['table_key'].nunique() if not TRENDS_DF.empty else 0} "
      f"tables; skipped {n_skipped} ragged rows")


def find_series(table_rx: str, series_rx: str, group_rx: Optional[str] = None) -> pd.Series:
    """Year-indexed values of one parsed series (first matching table/series), empty if not found."""
    if TRENDS_DF.empty:
        return pd.Series(dtype=float)
    m = TRENDS_DF["table_key"].str.contains(table_rx, case=False, regex=True) & \
        TRENDS_DF["series"].str.contains(series_rx, case=False, regex=True)
    if group_rx:
        m &= TRENDS_DF["group"].fillna("").str.contains(group_rx, case=False, regex=True)
    sub = TRENDS_DF[m]
    if sub.empty:
        return pd.Series(dtype=float)
    first = sub.groupby(["table_key", "series"], sort=False).ngroup() == 0
    return sub[first].set_index("year")["value"].sort_index()


KEY_SERIES = {
    "Real GDP growth (%)": find_series(r"Selected Economic Indicators", r"^Real GDP"),
    "CPI inflation, period average (%)": find_series(r"Selected Economic Indicators", r"^Period average", r"Consumer prices"),
    "CPI inflation, end of period (%)": find_series(r"Selected Economic Indicators", r"^End of period", r"Consumer prices"),
    "Credit to private sector growth (%)": find_series(r"Selected Economic Indicators", r"Credit to the private sector"),
    "Central government debt (% of GDP)": find_series(r"Selected Economic Indicators", r"^Central government debt"),
    "Bank NPL ratio (%)": find_series(r"Commercial Banks", r"^Nonperforming loans to total gross loans"),
    "Bank regulatory capital / RWA (%)": find_series(r"Commercial Banks", r"^Regulatory capital ratio"),
    "Credit union NPL ratio (%)": find_series(r"Credit Unions", r"^Nonperforming loans to total gross loans"),
    "Credit union regulatory capital (%)": find_series(r"Credit Unions", r"^Regulatory capital ratio"),
}
KEY_WIDE = pd.DataFrame(KEY_SERIES).T                       # wide: series × year
KEY_WIDE.columns = KEY_WIDE.columns.astype(int)
display(KEY_WIDE.round(2))

# %%
TREND_PROMPT = """The CONTEXT contains statistical tables from the IMF Article IV report on {country}.
Identify up to 8 time series most relevant to macro-financial stability trends (e.g. growth, inflation, credit, public debt,
bank and credit-union NPLs and capital). For each: name (row label as written), unit, source_table (table title as written),
theme (one of: {themes}; or "Growth & Outlook"), and points = every year/value pair shown in that row.
Copy numbers exactly; never interpolate. Use the row under the correct table title.

CONTEXT:
{context}"""
PROMPTS["TREND_PROMPT"] = TREND_PROMPT


class TrendPoint(BaseModel):
    year: int
    value: float


class TrendSeries(BaseModel):
    name: str
    unit: str
    source_table: str
    theme: str
    points: List[TrendPoint]


class TrendSet(BaseModel):
    series: List[TrendSeries]


trend_hits = []
for q in ["Selected Economic Indicators output prices money and credit", "Financial Stability Indicators commercial banks",
          "Financial Stability Indicators credit unions", "central government debt percent of GDP projected"]:
    trend_hits += VINDEX.search(q, k=3, kind="table")
TREND_CONTEXT = format_context(trend_hits, max_tokens=8000)
trend_res = llm.generate(TREND_PROMPT.format(country=C, themes="; ".join(THEMES), context=TREND_CONTEXT),
                         EXTRACTOR_MODEL, system=SYSTEM_ANALYST, schema=TrendSet, tag="trends")
LLM_TRENDS = trend_res.parsed.series if trend_res.ok else []


def _tok(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower())) - {"of", "the", "to", "in", "and", "percent", "total"}


def cross_check_trend(ts: TrendSeries) -> Dict[str, Any]:
    """Match an LLM series to the parsed table row with the highest name overlap (table title as tie-breaker)."""
    if TRENDS_DF.empty:
        return {"matched_series": None, "agreement": np.nan, "n_compared": 0}
    cands = TRENDS_DF[["table_key", "group", "series"]].drop_duplicates()
    def score(r):
        a, b = _tok(ts.name), _tok(r.series) | _tok(r.group or "")
        jac = len(a & b) / max(1, len(a | b))
        return jac + 0.25 * (len(_tok(ts.source_table) & _tok(r.table_key)) / max(1, len(_tok(ts.source_table))))
    best = max(cands.itertuples(), key=score)
    ref = TRENDS_DF[(TRENDS_DF.table_key == best.table_key) & (TRENDS_DF.series == best.series) &
                    (TRENDS_DF.group.fillna("") == (best.group or ""))].set_index("year")["value"]
    pairs = [(p.value, ref[p.year]) for p in ts.points if p.year in ref.index]
    agree = sum(abs(a - b) <= 0.051 for a, b in pairs)
    return {"matched_table": best.table_key, "matched_series": best.series, "n_compared": len(pairs),
            "agreement": agree / len(pairs) if pairs else np.nan}


TREND_CHECK = pd.DataFrame([{"llm_series": t.name, "llm_table": t.source_table, "theme": t.theme,
                             "n_points": len(t.points), **cross_check_trend(t)} for t in LLM_TRENDS])
display(TREND_CHECK)
if not TREND_CHECK.empty:
    print(f"LLM vs parser value agreement (mean over series): {TREND_CHECK['agreement'].mean():.0%}")
save_json({"parser": {"cells": len(TRENDS_DF), "skipped_ragged_rows": n_skipped,
                      "key_series": {k: {int(y): float(v) for y, v in s.items()} for k, s in KEY_SERIES.items()},
                      "projection_start_year": CONFIG["projection_start_year"]},
           "llm_series": [t.model_dump() for t in LLM_TRENDS],
           "cross_check": TREND_CHECK.to_dict(orient="records")}, "temporal_trends.json")
TRENDS_DF.to_csv(OUT_DIR / "parsed_table_cells.csv", index=False)
