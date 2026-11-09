# %% [markdown]
# ---
# ## Task 4 — Tidy transformation and the multi-plot dashboard
# ### 4.1 Wide → long with `pd.melt`
# Plotting libraries and grouped statistics both work best on **long (tidy)** data: one row per observation, with columns `series | year | value`. Every wide matrix the pipeline produced is melted here: indicator × year, core indicators, model benchmark × metric, judge scores × criterion, and theme × measure.

# %%
# (1) Time-series matrix: series × year  →  long
KEY_LONG = (pd.melt(KEY_WIDE.reset_index(names="series"), id_vars="series", var_name="year", value_name="value")
            .dropna(subset=["value"]).astype({"year": int}))
KEY_LONG["is_projection"] = KEY_LONG["year"] >= CONFIG["projection_start_year"]

# (2) Core indicator JSON (one wide row) → long, nulls kept so fallbacks stay visible
CORE_LONG = pd.melt(pd.DataFrame([CORE.model_dump()]), id_vars="country_name", var_name="indicator", value_name="value")
CORE_LONG["fallback"] = CORE_LONG["indicator"].map(lambda k: FALLBACKS.get(k, {}).get("fallback"))

# (3) Model benchmark: metrics in columns → long
BENCH_LONG = pd.melt(BENCH_DF, id_vars=["model", "style", "label", "latency_s"],
                     value_vars=["accuracy", "schema_compliant", "judge_support_rate"],
                     var_name="metric", value_name="score")
BENCH_LONG["score_pct"] = 100 * BENCH_LONG["score"].astype(float)

# (4) Judge scores: criteria in columns → long (section summaries + RAG answers)
SUM_LONG = pd.melt(SUMMARY_JUDGE_DF, id_vars=["id", "round"],
                   value_vars=["consistency", "completeness", "factual_alignment"],
                   var_name="criterion", value_name="score") if not SUMMARY_JUDGE_DF.empty else pd.DataFrame()
RAG_LONG = pd.melt(RAG_DF, id_vars=["q", "condition"], value_vars=["faithfulness", "completeness", "relevance"],
                   var_name="criterion", value_name="score") if not RAG_DF.empty else pd.DataFrame()

# (5) Themes: two measures → long
THEME_LONG = pd.melt(THEME_DF, id_vars="theme", value_vars=["lexicon_share_pct", "embedding_share_pct"],
                     var_name="measure", value_name="share_pct")

for name, df in [("KEY_LONG", KEY_LONG), ("CORE_LONG", CORE_LONG), ("BENCH_LONG", BENCH_LONG),
                 ("SUM_LONG", SUM_LONG), ("RAG_LONG", RAG_LONG), ("THEME_LONG", THEME_LONG)]:
    print(f"{name:<11} {df.shape}")
display(KEY_LONG.head(8))
KEY_LONG.to_csv(OUT_DIR / "key_series_long.csv", index=False)

# %% [markdown]
# ### 4.2 Dashboard
# Six panels, each answering one question:
#
# | Panel | Question | Form |
# |---|---|---|
# | A | Where does the report put its attention? | horizontal bars, lexicon vs embedding share |
# | B | How do growth, inflation and credit move over 2019–29? | lines, with projections shaded |
# | C | Does weak credit growth coincide with high NPLs? | scatter, banks vs credit unions, labelled by year |
# | D | Are bank and credit-union balance sheets getting healthier? | lines, 2013–23 |
# | E | Which model and prompt set-up extracts best? | grouped bars (%), latency annotated |
# | F | How much can the generated text be trusted? | box plot of judge scores by output type |
#
# Design rules: one y-axis per panel (no dual axes), every value in **percent units** (so no 0.04-vs-4% scale errors), a fixed colour-blind-safe categorical order, grid lines, units in the axis labels, and an executive text band at the bottom. Saved at **300 dpi**, together with every panel as its own file for the report.

# %%
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 12,
                     "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def legend_below(ax, ncol: int = 3):
    """Legends sit under the plot area so they never cover data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=ncol, frameon=False)


def no_data(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2, fontsize=12, transform=ax.transAxes, wrap=True)


def panel_themes(ax):
    d = THEME_DF.sort_values("lexicon_share_pct")
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["lexicon_share_pct"], height=0.38, color=PALETTE[0], label="Lexicon mentions (share %)")
    ax.barh(y - 0.2, d["embedding_share_pct"], height=0.38, color=PALETTE[1], label="Embedding-classified chunks (share %)")
    for yi, v, n in zip(y, d["lexicon_share_pct"], d["mentions_per_1k_words"]):
        ax.text(v + 0.5, yi + 0.2, f"{n:.1f}/1k words", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(y, d["theme"])
    ax.set_xlim(0, max(d[["lexicon_share_pct", "embedding_share_pct"]].max()) * 1.3 + 1)
    style_ax(ax, "A. Thematic density across the 7 financial-stability themes", "Share of total (%)")
    ax.legend(loc="lower right", frameon=False)


def panel_macro(ax):
    names = ["Real GDP growth (%)", "CPI inflation, period average (%)", "Credit to private sector growth (%)"]
    plotted = False
    for i, n in enumerate(names):
        s = KEY_SERIES.get(n, pd.Series(dtype=float))
        if s.empty:
            continue
        s = s[(s.index >= 2015)]
        ax.plot(s.index, s.values, color=PALETTE[i], lw=2.2, marker="o", ms=5, label=n)
        ax.annotate(f"{s.iloc[-1]:.1f}", (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK2)
        plotted = True
    if not plotted:
        return no_data(ax, "Selected Economic Indicators table not parsed")
    ax.axvspan(CONFIG["projection_start_year"] - 0.5, ax.get_xlim()[1], color="#f1f0ec", zorder=0)
    ax.text(CONFIG["projection_start_year"] - 0.3, 0.97, "IMF staff projections →", color=INK2, fontsize=10,
            va="top", transform=ax.get_xaxis_transform())
    ax.axhline(0, color=INK2, lw=0.8)
    style_ax(ax, "B. Growth, inflation and credit: history and projections",
             "Year   (no policy-rate series: Dominica uses the ECCB's quasi-currency board)", "Annual % change")
    legend_below(ax, ncol=3)


def panel_credit_npl(ax):
    credit = KEY_SERIES.get("Credit to private sector growth (%)", pd.Series(dtype=float))
    pairs = [("Commercial banks", "Bank NPL ratio (%)", PALETTE[0], "o"),
             ("Credit unions", "Credit union NPL ratio (%)", PALETTE[1], "s")]
    plotted = False
    for label, key, col, mk in pairs:
        npl = KEY_SERIES.get(key, pd.Series(dtype=float))
        yrs = sorted(set(credit.index) & set(npl.index))
        if not yrs:
            continue
        x, y = credit[yrs].values, npl[yrs].values
        ax.scatter(x, y, s=90, color=col, marker=mk, edgecolor="white", linewidth=1.5, label=label, zorder=3)
        for xi, yi, yr in zip(x, y, yrs):
            ax.annotate(str(yr), (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9.5, color=INK2)
        plotted = True
    if not plotted:
        return no_data(ax, "No overlapping years between credit growth and NPL tables")
    ax.axvline(0, color=INK2, lw=0.8)
    style_ax(ax, "C. Private credit growth vs NPL ratio (overlapping years)",
             "Credit to private sector, annual % change", "NPLs / gross loans (%)")
    legend_below(ax, ncol=2)


def panel_banking(ax):
    spec = [("Bank NPL ratio (%)", PALETTE[0], "-"), ("Credit union NPL ratio (%)", PALETTE[1], "-"),
            ("Bank regulatory capital / RWA (%)", PALETTE[0], "--"),
            ("Credit union regulatory capital (%)", PALETTE[1], "--")]
    plotted = False
    for name, col, ls in spec:
        s = KEY_SERIES.get(name, pd.Series(dtype=float))
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=col, ls=ls, lw=2.2, marker="o", ms=4.5, label=name)
        ax.annotate(f"{s.iloc[-1]:.1f}", (s.index[-1], s.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK2)
        plotted = True
    if not plotted:
        return no_data(ax, "Financial Stability Indicator tables not parsed")
    style_ax(ax, "D. Balance-sheet soundness: NPLs (solid) vs capital (dashed)", "Year", "Percent")
    legend_below(ax, ncol=2)


def panel_models(ax):
    if BENCH_DF.empty:
        return no_data(ax, "Benchmark not run")
    metrics = [("accuracy", "Accuracy vs human ground truth"), ("schema_compliant", "Schema compliance"),
               ("judge_support_rate", "Judge support rate")]
    labels = BENCH_DF["label"].tolist()
    x, w = np.arange(len(labels)), 0.26
    for i, (m, name) in enumerate(metrics):
        vals = 100 * BENCH_DF[m].astype(float).fillna(0).values
        ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=PALETTE[i], label=name)
    for xi, lat in zip(x, BENCH_DF["latency_s"]):
        ax.text(xi, 104, f"{lat:.1f}s", ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 115)
    style_ax(ax, "E. Indicator extraction: model × prompt benchmark", "", "Score (%)  ·  latency shown above bars")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)


def panel_judge(ax):
    groups = []
    if not SUMMARY_JUDGE_DF.empty:
        crit = ["consistency", "completeness", "factual_alignment"]
        first = SUMMARY_JUDGE_DF[SUMMARY_JUDGE_DF["round"] == 0]
        final = SUMMARY_JUDGE_DF.sort_values("round").groupby("id").tail(1)
        groups += [("Summaries\n(draft)", first[crit].mean(axis=1)), ("Summaries\n(final)", final[crit].mean(axis=1))]
    if not RAG_DF.empty:
        crit = ["faithfulness", "completeness", "relevance"]
        for cond, lab in [("closed_book", "RAG: closed\nbook"), ("single_query", "RAG: single\nquery"),
                          ("multi_query", "RAG: multi\nquery")]:
            groups.append((lab, RAG_DF[RAG_DF.condition == cond][crit].mean(axis=1)))
    groups = [(l, v.dropna()) for l, v in groups if len(v.dropna())]
    if not groups:
        return no_data(ax, "No judge scores available")
    bp = ax.boxplot([v.values for _, v in groups], patch_artist=True, widths=0.5,
                    medianprops={"color": INK, "lw": 2})
    for k, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[min(k, len(PALETTE) - 1)])
        patch.set_alpha(0.35)
    rng = np.random.default_rng(7)
    for k, (_, v) in enumerate(groups, start=1):
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v.values, s=26, color=INK2, zorder=3)
    ax.set_xticks(range(1, len(groups) + 1), [l for l, _ in groups])
    ax.set_ylim(0.8, 5.3)
    ax.axhline(CONFIG["judge_pass_threshold"], color=PALETTE[6], ls=":", lw=1.5)
    ax.text(len(groups) + 0.45, CONFIG["judge_pass_threshold"] + 0.06, "pass threshold", color=PALETTE[6],
            fontsize=9.5, ha="right")
    style_ax(ax, f"F. LLM-as-a-judge scores ({JUDGE_MODEL})", "", "Mean criterion score (1-5)")


def executive_text() -> str:
    def fmt(k, unit="%"):
        v = getattr(CORE, k)
        return f"{v:.1f}{unit}" if v is not None else "n/a"
    fb = ", ".join(f"{k.replace('_pct', '').replace('_', ' ')} ({v['fallback']})" for k, v in FALLBACKS.items()) or "none"
    top_theme = THEME_DF.iloc[0]["theme"]
    vul = "; ".join(i["title"] for i in RISK_INSIGHTS["vulnerabilities"][:3]) or "n/a"
    acc = MAIN_ACC["accuracy"]
    return (f"{C} — {CONFIG['imf_country_report_no']}  |  GDP growth forecast {fmt('gdp_growth_forecast_pct')}  ·  "
            f"Headline inflation {fmt('headline_inflation_rate_pct')}  ·  Bank NPL ratio {fmt('non_performing_loan_ratio')}  ·  "
            f"Unemployment {fmt('unemployment_rate_pct')}  ·  Policy rate {fmt('central_bank_policy_rate_pct')}  ·  "
            f"Tier 1 {fmt('tier_1_capital_ratio')}\n"
            f"Most-discussed theme: {top_theme}.   Key vulnerabilities: {vul}.\n"
            f"Fallbacks (null): {fb}.   Extraction accuracy vs human ground truth: "
            f"{'n/a' if pd.isna(acc) else f'{acc:.0%}'}  ·  judge support rate {MAIN_AUDIT['support_rate']:.0%}.")


PANELS = [("A_themes", panel_themes), ("B_macro", panel_macro), ("C_credit_vs_npl", panel_credit_npl),
          ("D_banking", panel_banking), ("E_model_benchmark", panel_models), ("F_judge_scores", panel_judge)]

fig = plt.figure(figsize=(20, 23))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.22], hspace=0.55, wspace=0.22, top=0.95)
for k, (_, fn) in enumerate(PANELS):
    fn(fig.add_subplot(gs[k // 2, k % 2]))
txt_ax = fig.add_subplot(gs[3, :])
txt_ax.axis("off")
txt_ax.text(0.0, 0.95, "Executive overview", fontsize=14, fontweight="bold", va="top")
txt_ax.text(0.0, 0.62, executive_text(), fontsize=11.5, va="top", linespacing=1.6,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f6f5f2", "edgecolor": GRID})
fig.suptitle(f"{C}: IMF 2024 Article IV — LLM-extracted macro-financial dashboard", fontsize=20,
             fontweight="bold", x=0.06, ha="left", y=0.99)
fig.text(0.06, 0.967, f"Extractor: {EXTRACTOR_MODEL}  ·  Judge: {JUDGE_MODEL}  ·  Embeddings: {CONFIG['embedding_model']}"
         f"  ·  Source: {CONFIG['imf_country_report_no']}, tables parsed from the PDF", fontsize=11, color=INK2)
DASHBOARD_PATH = FIG_DIR / "dashboard.png"
fig.savefig(DASHBOARD_PATH, dpi=300, bbox_inches="tight")
plt.show()

for name, fn in PANELS:                                     # individual high-resolution panels for the report
    f, ax = plt.subplots(figsize=(10, 6.8))
    fn(ax)
    f.tight_layout()
    f.savefig(FIG_DIR / f"panel_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(f)
print("Saved:", DASHBOARD_PATH, "and", len(PANELS), "individual panels (300 dpi)")

# %% [markdown]
# ---
# ## 5. Run summary and artefact index
# Every number below comes from the files in `outputs/`. These are the numbers the written report quotes.

# %%
pd.DataFrame(llm.log).to_csv(OUT_DIR / "llm_call_log.csv", index=False)
save_json(PROMPTS, "prompts.json")
log_df = pd.DataFrame(llm.log)
summary = {
    "LLM calls (total / served from cache / failed)":
        f"{len(log_df)} / {int(log_df['cached'].sum()) if len(log_df) else 0} / {int((~log_df['ok']).sum()) if len(log_df) else 0}",
    "Pages / sections summarised / chunks": f"{len(PAGES)} / {len(SECTION_SUMMARIES)} / {len(CHUNKS)}",
    "Section summaries within 100 words": f"{sum(s['within_limit'] for s in SECTION_SUMMARIES)}/{len(SECTION_SUMMARIES)}",
    "Mean judge score, summaries (draft → final)":
        (f"{SUMMARY_JUDGE_DF[SUMMARY_JUDGE_DF['round'] == 0][['consistency', 'completeness', 'factual_alignment']].mean().mean():.2f} → "
         f"{SUMMARY_JUDGE_DF.sort_values('round').groupby('id').tail(1)[['consistency', 'completeness', 'factual_alignment']].mean().mean():.2f}")
        if not SUMMARY_JUDGE_DF.empty else "n/a",
    "Exec-summary quotes verified": f"{sum(b['quote_verified'] for b in EXEC_SUMMARY['bullets'])}/{len(EXEC_SUMMARY['bullets'])}",
    "Insight quotes verified": f"{sum(i['quote_verified'] for k in ('growth_drivers', 'vulnerabilities') for i in RISK_INSIGHTS[k])}"
                               f"/{len(RISK_INSIGHTS['growth_drivers']) + len(RISK_INSIGHTS['vulnerabilities'])}",
    "Indicator accuracy vs ground truth": "n/a" if pd.isna(MAIN_ACC["accuracy"]) else f"{MAIN_ACC['accuracy']:.0%}",
    "Indicator fallbacks": ", ".join(FALLBACKS) or "none",
    "Mean RAG faithfulness (closed / single / multi)":
        " / ".join(f"{RAG_DF[RAG_DF.condition == c]['faithfulness'].mean():.2f}" for c in
                   ("closed_book", "single_query", "multi_query")) if not RAG_DF.empty else "n/a",
}
display(pd.DataFrame(summary.items(), columns=["Metric", "Value"]))
print("\nArtefacts:")
for p in sorted(OUT_DIR.glob("*")) + sorted(FIG_DIR.glob("*.png")):
    print(f"  {p.relative_to(BASE_DIR)}  ({p.stat().st_size / 1024:.0f} KB)")

# %% [markdown]
# ## 6. Human-oversight checklist (to complete before writing the report)
# The LLM steps are audited automatically, but the brief makes **you** responsible for verified accuracy. Before quoting any number:
# 1. Open the PDF at the page cited in `core_indicators.json → evidence` for each indicator, and confirm `ground_truth.json` against it. **p.36 especially**: check that the bank and credit-union rows have not been swapped.
# 2. Read every judge `unsupported_claims` entry in `summaries.json` and `rag_results.json`. Decide whether the judge was right; judges produce false positives too.
# 3. Look at the rows where `judge_agrees_with_truth` is `False` in the audit table (§2.4). Those are cases of judge error or judge bias worth discussing.
# 4. Check the section table in §1.3 against the report's table of contents (pp. 6–7).
# 5. Make sure every figure you put in the report is in `figures/` at 300 dpi and is referenced in the text.
