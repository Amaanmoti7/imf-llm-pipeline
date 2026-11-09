"""Offline end-to-end smoke test: executes the real notebook with a mocked Gemini backend and a hashing
embedder (no network). Usage: python tests/run_smoke.py <workdir> <pdf>"""
import os, sys, shutil, json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
work, pdf = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if work.exists():
    shutil.rmtree(work)
(work / "data").mkdir(parents=True)
shutil.copy(pdf, work / "data" / "dominica_2024_article_iv.pdf")
if (ROOT / "ground_truth.json").exists():
    shutil.copy(ROOT / "ground_truth.json", work / "ground_truth.json")

MOCK_EMBEDDER = r'''
import numpy as _np, re as _re, hashlib as _h
class SentenceTransformer:  # offline stand-in: hashed bag-of-words embeddings
    def __init__(self, name, *a, **k): self.dim = 384
    def get_sentence_embedding_dimension(self): return self.dim
    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, **k):
        out = _np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in _re.findall(r"[a-z]{3,}", t.lower()):
                out[i, int(_h.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = _np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
        return out / n
'''

MOCK_LLM = r'''
import json as _json, re as _re, hashlib as _hl
def _src_sentences(prompt):
    body = prompt.split("<<<")[-1] if "<<<" in prompt else prompt
    s = [x.strip() for x in _re.split(r"(?<=[.])\s+", body) if len(x.split()) > 12]
    return s or ["Dominica's output has recovered to its pre-pandemic level"]
def _mock_call(self, model, prompt, system, schema, temperature):
    h = int(_hl.md5(prompt.encode()).hexdigest(), 16)
    name = schema.__name__ if schema is not None else None
    sents = _src_sentences(prompt)
    q = lambda k: " ".join(sents[k % len(sents)].split()[:18])
    if name == "SummaryJudgement":
        lo = 3 if (h % 3 == 0 and "Previous draft" not in prompt) else 4
        return _json.dumps({"consistency": 5, "completeness": lo, "factual_alignment": 4 + h % 2,
                            "unsupported_claims": [] if lo > 3 else ["claim X"], "missing_points": ["point Y"],
                            "verdict": "pass" if lo > 3 else "revise", "feedback": "Add the debt figure."})
    if name == "ExecutiveSummary":
        return _json.dumps({"bullets": [{"point": f"Finding {i} about growth and debt.", "theme": "Sovereign Debt",
                                         "evidence_quote": q(i) if i % 4 else "invented quote that is not in the source at all"} for i in range(8)]})
    if name == "RiskInsights":
        mk = lambda p: [{"title": f"{p} {i}", "explanation": "Explanation.", "theme": "Sovereign Debt", "evidence_quote": q(i)} for i in range(6)]
        return _json.dumps({"growth_drivers": mk("Driver"), "vulnerabilities": mk("Vulnerability")})
    if name == "IndicatorExtraction":
        iv = lambda v, s="extracted", **k: {"value": v, "reference_year": "2023", "status": s, "source_quote": "q",
                                           "source_chunk": "C001", "note": "", **k}
        lite = "lite" in model
        return _json.dumps({"country_name": "Dominica", "gdp_growth_forecast_pct": iv(4.6),
            "headline_inflation_rate_pct": iv(3.5), "central_bank_policy_rate_pct": iv(None, "not_reported"),
            "non_performing_loan_ratio": iv(13.41 if lite else 12.9), "unemployment_rate_pct": iv(23.0),
            "tier_1_capital_ratio": iv(None, "not_reported", proxy_value=27.51, proxy_definition="total capital/RWA")})
    if name == "IndicatorAudit":
        return _json.dumps({"verdicts": [{"indicator": k, "verdict": "supported"} for k in
            ["gdp_growth_forecast_pct", "headline_inflation_rate_pct", "non_performing_loan_ratio", "unemployment_rate_pct"]] +
            [{"indicator": "central_bank_policy_rate_pct", "verdict": "correct_null"},
             {"indicator": "tier_1_capital_ratio", "verdict": "correct_null"}], "overall_score": 4})
    if name == "TrendSet":
        return _json.dumps({"series": [{"name": "Real GDP", "unit": "%", "source_table": "Table 1. Selected Economic Indicators",
            "theme": "Growth & Outlook", "points": [{"year": 2019 + i, "value": v} for i, v in enumerate([5.5, -16.6, 6.9, 5.6, 4.7])]},
            {"name": "Nonperforming loans to total gross loans", "unit": "%", "source_table": "Table 5a Commercial Banks",
             "theme": "Banking Capital & Liquidity", "points": [{"year": 2022, "value": 13.8}, {"year": 2023, "value": 12.9}]}]})
    if name == "QueryVariants":
        return _json.dumps({"queries": ["variant one about debt", "variant two on provisioning", "variant three inflation"]})
    if name == "RAGJudgement":
        f = 2 if "own knowledge" in prompt else 4 + h % 2
        return _json.dumps({"faithfulness": f, "completeness": 3 + h % 3, "relevance": 5, "unsupported_claims": [], "rationale": "ok"})
    if "return them as JSON" in prompt:
        return '```json\n{"country_name": "Dominica", "gdp_growth_forecast_pct": "4.6%", "headline_inflation_rate_pct": 3.5, "non_performing_loan_ratio": 13.41, "unemployment_rate_pct": null, "tier_1_capital_ratio": null}\n```'
    if "Rewrite the summary" in prompt:
        return "Compressed summary of the section with key figures."
    if "own knowledge" in prompt:
        return "Dominica's central bank raised rates to 5 percent in 2023."
    if "CONTEXT" in prompt:
        return "Staff recommend stepping up provisioning [C010]."
    words = ("Dominica growth recovered strongly while public debt stays high " * (20 if h % 5 == 0 else 7)).split()
    return " ".join(words)
LLMClient._call_api = _mock_call
llm.min_interval = 0.0
'''

nb = nbformat.read(ROOT / "IMF_Dominica_LLM_Pipeline.ipynb", as_version=4)
cells = []
for c in nb.cells:
    if c.cell_type == "code" and c.source.lstrip().startswith("%pip"):
        continue
    cells.append(c)
    if c.cell_type == "code" and "from sentence_transformers import SentenceTransformer" in c.source:
        cells.append(nbformat.v4.new_code_cell(MOCK_EMBEDDER))
    if c.cell_type == "code" and "class LLMClient" in c.source:
        cells.append(nbformat.v4.new_code_cell(MOCK_LLM))
nb.cells = cells
os.environ["GEMINI_API_KEY"] = "mock-key"
os.environ["IMF_PIPELINE_DIR"] = str(work)
client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(work)}})
import re as _re
try:
    client.execute()
except Exception as exc:
    msg = _re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    print("\n".join(msg.splitlines()[-25:]))
    raise SystemExit(1)
finally:
    nbformat.write(nb, work / "executed_smoke.ipynb")
print("SMOKE TEST PASSED")
