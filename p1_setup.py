# %% [markdown]
# # IMF Article IV PDF → Dashboard: an LLM extraction, evaluation and RAG pipeline
#
# **Report analysed:** *Dominica: 2024 Article IV Consultation – Press Release; Staff Report; and Statement by the Executive Director for Dominica* (IMF Country Report No. 24/192, June 2024).
# **Allocation:** SRN 24160777 → Dominica, 2023–2024.
#
# This notebook runs end-to-end and builds:
#
# | Task | What the notebook does | Main outputs |
# |---|---|---|
# | 1 | PDF ingestion (`pypdf`), cleaning, section detection, token-aware chunking, section summaries (<100 words), executive summary, **dual-LLM judge + refinement loop** | `outputs/sections.json`, `outputs/summaries.json`, `outputs/executive_summary.json` |
# | 2 | 7-theme distribution (lexicon + embedding classifier), strengths/vulnerabilities, **schema-constrained JSON indicators** with fallbacks, temporal trends, judge audit | `outputs/themes.json`, `outputs/risk_insights.json`, `outputs/core_indicators.json`, `outputs/indicator_audit.json`, `outputs/temporal_trends.json` |
# | 3 | Local RAG: `sentence-transformers` embeddings + FAISS `IndexFlatL2`, grounded Q&A, closed-book hallucination baseline | `outputs/rag_results.json` |
# | 4 | `pd.melt` tidy transforms + a 6-panel, 300-dpi dashboard | `figures/dashboard.png` + individual panels |
# | Extension | **(a)** model and system-instruction benchmarking, **(b)** Multi-Query RAG vs single-query RAG | `outputs/benchmark.json`, `outputs/rag_results.json` |
#
# ### How to run
# 1. **Runtime → Run all.** In Colab, add your Gemini key under **🔑 Secrets** as `GEMINI_API_KEY` and enable notebook access.
# 2. **No key? It still runs.** Every LLM response is cached on disk (`cache/llm/`). When the cache is present (it is committed to the GitHub repo), the notebook replays the cached responses with no API key and no network calls. Set `CONFIG["repo_url"]` so that Colab clones the cache automatically.
# 3. The PDF is downloaded from imf.org. If that fails, Colab asks you to upload it.
#
# > Design principle: the LLM is never trusted blindly. Every generated artefact is (i) schema-validated, (ii) audited by a *second* LLM acting as judge, and (iii) where possible checked deterministically (quote verification, table-parser cross-checks, human ground truth).

# %% [markdown]
# ## 0. Setup
# Installs the packages Colab doesn't ship with. `pypdf` handles parsing, `google-genai` is the Gemini SDK, `sentence-transformers` + `faiss-cpu` power local RAG, and `tiktoken` is the fallback tokenizer.

# %%
%pip install -q "pypdf>=4.2" "google-genai>=1.20" "sentence-transformers>=3.0" "faiss-cpu>=1.8" "tiktoken>=0.7"

# %%
# ---- Standard library ------------------------------------------------------
import os, re, sys, json, time, math, hashlib, unicodedata, warnings, subprocess, textwrap
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter, defaultdict

# ---- Third party -----------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types as genai_types
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
IN_COLAB = "google.colab" in sys.modules
print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | running in Colab: {IN_COLAB}")

# %% [markdown]
# ### 0.1 Configuration
# All tunable parameters live in one dictionary, so the notebook body has no magic numbers. The chunk sizes follow the brief (300–500 tokens, 50-token overlap). The embedding model is `BAAI/bge-small-en-v1.5`, which has a **512-token** input window, so a 400-token chunk is embedded whole. (`all-MiniLM-L6-v2` would silently truncate anything past 256 tokens.)
#
# Model names are *preferences*. The notebook asks the Gemini API which models are actually available and picks the first match. The resolved names are saved to `outputs/run_config.json` so that cached replays use exactly the same models.

# %%
CONFIG: Dict[str, Any] = {
    # --- Assignment allocation ---------------------------------------------
    "student_id": "24160777",
    "country": "Dominica",
    "confusable_country": "the Dominican Republic",   # explicit guard used in prompts
    "report_period": "2023-2024",
    "report_title": ("Dominica: 2024 Article IV Consultation-Press Release; Staff Report; "
                     "and Statement by the Executive Director for Dominica"),
    "imf_country_report_no": "IMF Country Report No. 24/192",
    "pdf_url": "https://www.imf.org/-/media/files/publications/cr/2024/english/1dmaea2024001-print-pdf.pdf",
    "pdf_filename": "dominica_2024_article_iv.pdf",
    "repo_url": "",                 # e.g. "https://github.com/<user>/imf-llm-pipeline" (enables key-less replay in Colab)

    # --- Chunking -----------------------------------------------------------
    "chunk_target_tokens": 400,
    "chunk_overlap_tokens": 50,
    "chunk_min_tokens": 60,
    "chunk_hard_max_tokens": 500,   # below the embedder's 512-token window

    # --- Retrieval ------------------------------------------------------------
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_query_prefix": "Represent this sentence for searching relevant passages: ",
    "retrieval_k": 5,
    "multi_query_n_variants": 3,
    "multi_query_top_k": 8,
    "rrf_k": 60,

    # --- Summarisation / judging -------------------------------------------
    "summary_word_limit": 100,      # brief: "fewer than 100 words"
    "summary_target_words": 85,     # prompt target leaves a safety margin
    "section_max_tokens": 12_000,   # longer sections are summarised map-reduce style
    "exec_summary_max_input_tokens": 120_000,
    "judge_pass_threshold": 4,      # every criterion must be >= 4 out of 5
    "max_refinement_rounds": 1,

    # --- Time series --------------------------------------------------------
    "projection_start_year": 2024,  # Table 1 marks 2023 as "Est." and 2024+ as "Projected"

    # --- LLM ----------------------------------------------------------------
    "temperature": None,            # None = model default (recommended for Gemini 3.x); reproducibility comes from the cache
    "min_seconds_between_calls": 4.0,   # free-tier friendly throttle
    "max_retries": 6,
    "extractor_model_preferences": ["gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash", "gemini-2.0-flash"],
    "judge_model_preferences":     ["gemini-3.8-flash", "gemini-3.5-pro", "gemini-3-pro", "gemini-2.5-pro",
                                    "gemini-3-flash", "gemini-2.5-flash"],
    "lite_model_preferences":      ["gemini-3.5-flash-lite", "gemini-3-flash-lite", "gemini-2.5-flash-lite",
                                    "gemini-2.0-flash-lite"],
}

THEMES = ["Inflation & Prices", "Interest Rates & Credit", "Housing & Real Estate",
          "Banking Capital & Liquidity", "Sovereign Debt", "Labor Market Dynamics",
          "Systemic Financial Risk"]


def resolve_base_dir() -> Path:
    """Work in the repo folder if we are in one; in Colab, optionally clone the repo to get the cache."""
    env = os.environ.get("IMF_PIPELINE_DIR")
    if env:
        return Path(env).resolve()
    if IN_COLAB and CONFIG["repo_url"]:
        target = Path("/content") / Path(CONFIG["repo_url"].rstrip("/")).stem
        if not target.exists():
            subprocess.run(["git", "clone", "--depth", "1", CONFIG["repo_url"], str(target)], check=False)
        if target.exists():
            return target
    return Path.cwd()


BASE_DIR = resolve_base_dir()
DATA_DIR, CACHE_DIR = BASE_DIR / "data", BASE_DIR / "cache"
OUT_DIR, FIG_DIR = BASE_DIR / "outputs", BASE_DIR / "figures"
for d in (DATA_DIR, CACHE_DIR / "llm", CACHE_DIR / "embeddings", OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
PDF_PATH = DATA_DIR / CONFIG["pdf_filename"]
print("Working directory:", BASE_DIR)


def save_json(obj: Any, name: str) -> Path:
    """Write a machine-readable artefact to outputs/ (numpy types are converted)."""
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, BaseModel): return o.model_dump()
        if isinstance(o, Path): return str(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")
    path = OUT_DIR / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
    return path

# %% [markdown]
# ### 0.2 LLM client: disk cache, throttling, retries, schema parsing
# One wrapper handles every Gemini call. It:
# * **Caches** each response on disk under a SHA-256 key of (model, system instruction, prompt, schema, temperature). Re-runs cost no quota and replay deterministically, cached latency included, so benchmark numbers stay stable.
# * **Throttles** calls to respect free-tier rate limits, and **retries** transient errors (429/5xx) with exponential backoff.
# * **Enforces schemas**. When a Pydantic model is passed, Gemini runs in JSON mode with `response_schema`, and the reply is validated again locally. Validation failures are *recorded* rather than raised, so that schema compliance can be measured.
#
# The API key comes from Colab Secrets or the environment. It is never printed or written to disk.

# %%
def get_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and IN_COLAB:
        try:
            from google.colab import userdata
            key = userdata.get("GEMINI_API_KEY")
        except Exception:
            key = None
    return key or None


@dataclass
class LLMResult:
    text: str
    parsed: Any
    model: str
    latency_s: float
    cached: bool
    tag: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def extract_json_block(text: str) -> Optional[Any]:
    """Robustly parse JSON from a model reply (handles ```json fences and leading prose)."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


class LLMClient:
    TRANSIENT = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                 "DEADLINE", "overloaded", "timed out", "Timeout")

    def __init__(self, api_key: Optional[str], cache_dir: Path,
                 min_interval: float = 4.0, max_retries: int = 6):
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_call = 0.0
        self.log: List[Dict[str, Any]] = []

    # -- low-level API call (isolated so it can be swapped/mocked) -----------
    def _call_api(self, model: str, prompt: str, system: Optional[str],
                  schema: Optional[type], temperature: Optional[float]) -> str:
        cfg: Dict[str, Any] = {}
        if system:
            cfg["system_instruction"] = system
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
        if temperature is not None:
            cfg["temperature"] = temperature
        resp = self.client.models.generate_content(
            model=model, contents=prompt, config=genai_types.GenerateContentConfig(**cfg))
        return resp.text or ""

    def _cache_key(self, model, prompt, system, schema, temperature) -> str:
        payload = {"model": model, "system": system, "prompt": prompt, "temperature": temperature,
                   "schema": schema.model_json_schema() if schema is not None else None}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def generate(self, prompt: str, model: str, system: Optional[str] = None,
                 schema: Optional[type] = None, temperature: Optional[float] = CONFIG["temperature"],
                 tag: str = "") -> LLMResult:
        key = self._cache_key(model, prompt, system, schema, temperature)
        path = self.cache_dir / f"{key}.json"
        text, latency, cached, error = "", 0.0, False, None

        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            text, latency, cached = rec["text"], rec["latency_s"], True
        elif self.client is None:
            error = ("No API key available and this prompt is not in the cache. "
                     "Set GEMINI_API_KEY (Colab Secrets) or clone the repo that contains cache/llm/.")
        else:
            for attempt in range(self.max_retries):
                wait = self.min_interval - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                try:
                    t0 = time.perf_counter()
                    self._last_call = time.time()
                    text = self._call_api(model, prompt, system, schema, temperature)
                    latency = time.perf_counter() - t0
                    error = None
                    break
                except Exception as exc:                     # noqa: BLE001 - we classify below
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    if not any(t in str(exc) for t in self.TRANSIENT):
                        break                                 # permanent error: don't burn retries
                    time.sleep(min(60, 5 * 2 ** attempt))
            if error is None:
                path.write_text(json.dumps({"tag": tag, "model": model, "latency_s": latency,
                                            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "prompt_preview": prompt[:300], "text": text},
                                           ensure_ascii=False, indent=1), encoding="utf-8")

        parsed = None
        if schema is not None and error is None:
            raw = extract_json_block(text)
            try:
                parsed = schema.model_validate(raw)
            except (ValidationError, TypeError) as exc:
                error = f"SchemaValidationError: {str(exc)[:300]}"
        self.log.append({"tag": tag, "model": model, "latency_s": round(latency, 3), "cached": cached,
                         "ok": error is None, "error": error})
        return LLMResult(text=text, parsed=parsed, model=model, latency_s=latency,
                         cached=cached, tag=tag, error=error)


API_KEY = get_api_key()
llm = LLMClient(API_KEY, CACHE_DIR / "llm", CONFIG["min_seconds_between_calls"], CONFIG["max_retries"])
print("Gemini API key found:", API_KEY is not None,
      "| cached responses:", len(list((CACHE_DIR / "llm").glob("*.json"))))

# %% [markdown]
# ### 0.3 Model resolution (dual-LLM set-up)
# * **Extractor/summariser:** a Gemini *Flash* model, chosen for speed and cost.
# * **Judge:** a *different* model, preferably a stronger or newer one. A model grading its own output shows **self-preference bias** (Zheng et al., 2023), so the judge is always a different model whenever one is available.
# * **Lite:** a Flash-Lite variant, used only in the benchmarking extension.

# %%
def list_available_models(client) -> List[str]:
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        names.append(m.name.split("/")[-1])
    return names


def pick_model(prefs: List[str], available: List[str], exclude: Tuple[str, ...] = ()) -> Optional[str]:
    """Exact match first, then the first available name that starts with the preference
    (e.g. 'gemini-2.5-flash' matches 'gemini-2.5-flash-001'), never mixing in lite or special variants."""
    banned = ("image", "tts", "audio", "live", "embedding", "vision")
    for p in prefs:
        if p in available and p not in exclude:
            return p
        for a in sorted(available, reverse=True):
            if a.startswith(p) and a not in exclude and not any(b in a for b in banned) \
                    and ("lite" in p or "lite" not in a):
                return a
    return None


RUN_CONFIG_PATH = OUT_DIR / "run_config.json"
if llm.client is not None:
    try:
        AVAILABLE = list_available_models(llm.client)
    except Exception as exc:                                  # noqa: BLE001
        print("Could not list models (", exc, ") – using first preferences.")
        AVAILABLE = []
    if AVAILABLE:
        EXTRACTOR_MODEL = pick_model(CONFIG["extractor_model_preferences"], AVAILABLE)
        JUDGE_MODEL = pick_model(CONFIG["judge_model_preferences"], AVAILABLE, exclude=(EXTRACTOR_MODEL,)) \
            or EXTRACTOR_MODEL
        LITE_MODEL = pick_model(CONFIG["lite_model_preferences"], AVAILABLE)
    else:
        EXTRACTOR_MODEL = CONFIG["extractor_model_preferences"][0]
        JUDGE_MODEL = CONFIG["judge_model_preferences"][0]
        LITE_MODEL = CONFIG["lite_model_preferences"][0]
    save_json({"extractor_model": EXTRACTOR_MODEL, "judge_model": JUDGE_MODEL, "lite_model": LITE_MODEL,
               "config": CONFIG, "resolved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, "run_config.json")
elif RUN_CONFIG_PATH.exists():                                # key-less replay from cache
    _rc = json.loads(RUN_CONFIG_PATH.read_text())
    EXTRACTOR_MODEL, JUDGE_MODEL, LITE_MODEL = _rc["extractor_model"], _rc["judge_model"], _rc["lite_model"]
else:
    raise RuntimeError("No GEMINI_API_KEY and no outputs/run_config.json cache – cannot run the LLM steps.")

if EXTRACTOR_MODEL is None:
    raise RuntimeError("No Gemini Flash model is available for this key. Edit CONFIG['extractor_model_preferences'].")
print(f"Extractor: {EXTRACTOR_MODEL} | Judge: {JUDGE_MODEL} | Lite (benchmark): {LITE_MODEL}")
if JUDGE_MODEL == EXTRACTOR_MODEL:
    print("⚠️  Judge is the same model as the extractor – self-preference bias is likely; discuss in the report.")
