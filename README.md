# IMF Article IV PDF → Dashboard: LLM extraction, evaluation and RAG

An end-to-end Python pipeline (Google Colab) that turns an unstructured IMF Article IV Country Report into structured indicators, thematic distributions, risk insights, grounded Q&A and a 300-dpi analytics dashboard, with an LLM-as-a-judge evaluation loop at every step.

**Document:** *Dominica: 2024 Article IV Consultation – Press Release; Staff Report; and Statement by the Executive Director for Dominica*, IMF Country Report No. 24/192 (June 2024). Allocation: Dominica, 2023–2024.

## Pipeline

```
PDF ──pypdf (plain + layout)──► cleaning (unicode, learned headers/footers, reflow)
    ──► parts & sections ──► token-aware chunks (400 tok, 50 overlap; tables keep title/header)
    ──► bge-small embeddings ──► FAISS IndexFlatL2
         │
         ├─ Task 1  section summaries (<100 words) + executive summary ──► judge ──► refine ──► re-judge
         ├─ Task 2  7-theme distribution · strengths/vulnerabilities · RAG-grounded JSON indicators
         │          (Pydantic response_schema, null fallbacks with reasons) · table parser trends
         │          · judge audit + human ground truth
         ├─ Task 3  closed-book vs single-query RAG vs Multi-Query RAG (RRF) · judge on shared evidence
         ├─ Ext.    Flash vs Flash-Lite × strict vs generic prompt (compliance, latency, accuracy, judge)
         └─ Task 4  pd.melt tidy frames ──► 6-panel dashboard (300 dpi) + individual panels
```

## Running it

1. Open `IMF_Dominica_LLM_Pipeline.ipynb` in Google Colab.
2. Add a Gemini API key in **Secrets** (🔑) as `GEMINI_API_KEY`, then enable notebook access.
3. **Runtime → Run all.** The PDF downloads from imf.org automatically; if that fails, you are asked to upload it.

**Running without an API key (markers):** every Gemini response is cached in `cache/llm/`, and the resolved model names are saved in `outputs/run_config.json`. With `CONFIG["repo_url"]` set to this repository, Colab clones the repo and the notebook replays the cached responses with **no key and no API calls**. The outputs are identical.

## Repository layout

| Path | Contents |
|---|---|
| `IMF_Dominica_LLM_Pipeline.ipynb` | The executable notebook (all tasks, prompts, figures) |
| `ground_truth.json` | Human-verified reference values for the six indicators (with PDF page references) |
| `outputs/` | Machine-readable results: `core_indicators.json`, `indicator_audit.json`, `themes.json`, `risk_insights.json`, `summaries.json`, `executive_summary.json`, `temporal_trends.json`, `rag_results.json`, `benchmark.json`, `prompts.json`, `llm_call_log.csv`, … |
| `figures/` | `dashboard.png` and six individual panels, all at 300 dpi |
| `cache/` | LLM response cache and embedding cache (enables key-less replay) |
| `src_parts/`, `tools/build_notebook.py` | Notebook source in percent-format, and the script that assembles the `.ipynb` |
| `tests/` | Offline smoke test: synthetic IMF-style PDF + mocked Gemini backend |

## Models and libraries

* Google Gemini Flash / Flash-Lite via the `google-genai` SDK (extractor, judge, benchmark)
* `BAAI/bge-small-en-v1.5` sentence embeddings (Xiao et al., 2023) via `sentence-transformers` (Reimers & Gurevych, 2019)
* FAISS `IndexFlatL2` (Johnson, Douze & Jégou, 2019)
* `pypdf`, `pandas`, `matplotlib`, `pydantic`

## Development

```bash
python tools/build_notebook.py                         # src_parts/*.py -> notebook
python tests/make_test_pdf.py /tmp/test.pdf            # synthetic report with the same traps
python tests/run_smoke.py /tmp/smoke /tmp/test.pdf     # executes the notebook offline (mock LLM)
```

## License

MIT. See `LICENSE`. The IMF report itself is not redistributed; the notebook downloads it from imf.org.
