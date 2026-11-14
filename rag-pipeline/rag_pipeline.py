"""
RAG + QuickStats tools for the silage-vs-grain critic (Hack the Harvest).

What lives here
  1. Ingest: state reports (pdf / txt / md / html) -> chunks with metadata (state, year, report type, page)
  2. Index:  embeddings saved to disk (numpy, no FAISS needed at this corpus size)
  3. Tools the agents call:
       retrieve_reports(query, state, year, k)       -> passages + citations   (unstructured text)
       silage_share_history(state, county, years)    -> numbers from QuickStats (structured, never from RAG)
       check_prediction(state, pred_silage, pred_grain, ...) -> deterministic plausibility check
  4. run_critic(): Nemotron tool-calling loop that turns checks + passages into flags JSON

Rule: numbers come from QuickStats and our models. RAG only supplies context and citations.

CLI (run from anywhere; paths are relative to this file)
  python rag-pipeline/rag_pipeline.py build
  python rag-pipeline/rag_pipeline.py ask "When is corn silage usually harvested?" --state Maryland
  python rag-pipeline/rag_pipeline.py share --state Maryland
  python rag-pipeline/rag_pipeline.py check --state Maryland --silage 60000 --grain 380000

Layout
  rag-pipeline/
    rag_pipeline.py
    dataset/handbooks/   usual planting + harvesting dates, other national references
    dataset/national/    weekly national Crop Progress (US_...)
    dataset/maryland/    Maryland weekly Crop Progress & Condition (MD_...)
    dataset/quickstats/  corn_grain_silage_acres_2012_2025.csv
    rag_index/           built by `build`, not committed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAG_DIR = Path(__file__).resolve().parent            # .../FieldVision/rag-pipeline
REPO_DIR = RAG_DIR.parent                               # .../FieldVision
DATASET_DIR = Path(os.environ.get("RAG_DATASET", RAG_DIR / "dataset"))
QUICKSTATS_CSV = Path(os.environ.get(
    "QUICKSTATS_CSV", DATASET_DIR / "quickstats" / "corn_grain_silage_acres_2012_2025.csv"))
INDEX_DIR = Path(os.environ.get("RAG_INDEX", RAG_DIR / "rag_index"))

# Embedding backend: "local" (sentence-transformers on our GPU, free),
# "nvidia" (build.nvidia.com embeddings, uses Brev credits), or "hash" (tests only).
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "local")
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
NV_EMBED_MODEL = os.environ.get("NV_EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")  # confirm with client.models.list()
NV_BASE_URL = "https://integrate.api.nvidia.com/v1"
CHAT_MODEL = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b")

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
NAME_TO_STATE = {v.lower(): v for v in STATES.values()}


# ---------------------------------------------------------------- 1. ingest

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    state: str          # full state name, or "US" for national reports
    year: int | None
    report_type: str    # crop_progress | crop_production | acreage | other


def _read_pages(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz  # pymupdf
        with fitz.open(path) as doc:
            return [page.get_text("text", sort=True) for page in doc]
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
    return [raw]


def infer_state(path: Path, text: str) -> str:
    stem = re.sub(r"[_\-\.]+", " ", path.stem)
    if stem.split() and stem.split()[0].upper() == "US":  # national reports: name them US_...
        return "US"
    for token in stem.split():
        if token.upper() in STATES and len(token) == 2:
            return STATES[token.upper()]
    low = stem.lower()
    for name_low, name in NAME_TO_STATE.items():
        if name_low in low:
            return name
    if "national" in {p.lower() for p in path.parts} or "handbooks" in {p.lower() for p in path.parts}:
        return "US"
    # a document that names many states near the top is national (handbooks, US summaries)
    head = text[:3000].lower()
    counts = {name: head.count(name_low) for name_low, name in NAME_TO_STATE.items()}
    if sum(1 for c in counts.values() if c) >= 5:
        return "US"
    best = max(counts, key=counts.get)
    return best if counts[best] >= 2 else "US"


def infer_year(path: Path, text: str) -> int | None:
    m = re.search(r"(20[0-3]\d)", path.stem) or re.search(r"(20[0-3]\d)", text[:2000])
    return int(m.group(1)) if m else None


def infer_report_type(path: Path, text: str) -> str:
    s = (path.stem + " " + text[:500]).lower()
    if "progress" in s:
        return "crop_progress"
    if "production" in s:
        return "crop_production"
    if "acreage" in s:
        return "acreage"
    return "other"


_STATE_HEADER = re.compile(r"Dates\s+by\s+Crop\s*[\u2013\u2014-]\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)")


def _lookup_state(name: str) -> str | None:
    words = name.split()
    for n in (" ".join(words[:2]), words[0]):  # "Maryland State" -> Maryland, "West Virginia" stays whole
        if n.lower() in NAME_TO_STATE:
            return NAME_TO_STATE[n.lower()]
    return None


def _single_state(text: str) -> str:
    """The one state a passage names, matched as whole words; 'US' if none or several."""
    low = text.lower()
    found = {name for name_low, name in NAME_TO_STATE.items() if re.search(rf"\b{re.escape(name_low)}\b", low)}
    for longer, shorter in (("West Virginia", "Virginia"), ("New Mexico", "Mexico")):
        if longer in found:
            found.discard(shorter)
    return found.pop() if len(found) == 1 else "US"


def state_segments(page_text: str, doc_state: str) -> list[tuple[str, str]]:
    """Split a page into (state, text) pieces.

    National documents print state tables back to back ('Usual Planting and Harvesting Dates by
    Crop - Maryland'), often two to a page. Each table gets its own state tag so a Maryland query
    finds Maryland's table and not Missouri's. Text before the first header on a page, and pages
    with no header, are tagged by the single state they name, else 'US'.
    """
    if doc_state != "US":
        return [(doc_state, page_text)]
    heads = list(_STATE_HEADER.finditer(page_text))
    if not heads:
        return [(_single_state(page_text[:400]), page_text)]
    out = []
    if page_text[:heads[0].start()].strip():
        out.append(("US", page_text[:heads[0].start()]))  # tail of the previous page's table
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(page_text)
        out.append((_lookup_state(m.group(1)) or "US", page_text[m.start():end]))
    return out


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf}\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        while len(p) > size:  # a single huge paragraph or a table dump
            chunks.append(p[:size])
            p = p[size - overlap:]
        buf = (chunks[-1][-overlap:] + "\n" + p) if chunks else p
    if buf:
        chunks.append(buf)
    return chunks


def load_reports(report_dir: Path) -> list[Chunk]:
    out: list[Chunk] = []
    files = [p for p in sorted(report_dir.rglob("*"))
             if p.suffix.lower() in {".pdf", ".txt", ".md", ".html", ".htm"}]
    for path in files:
        pages = _read_pages(path)
        full = "\n".join(pages)
        state, year, rtype = infer_state(path, full), infer_year(path, full), infer_report_type(path, full)
        for page_no, page_text in enumerate(pages, start=1):
            i = 0
            for seg_state, seg_text in state_segments(page_text, state):
                for piece in split_text(seg_text):
                    cid = hashlib.md5(f"{path.name}:{page_no}:{i}".encode()).hexdigest()[:12]
                    out.append(Chunk(cid, piece, path.name, page_no, seg_state, year, rtype))
                    i += 1
    return out


# ---------------------------------------------------------------- 2. embed + index

class Embedder:
    def __init__(self, backend: str = EMBED_BACKEND):
        self.backend = backend
        if backend == "local":
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(LOCAL_EMBED_MODEL, device=device)
        elif backend == "nvidia":
            from openai import OpenAI
            self.client = OpenAI(base_url=NV_BASE_URL, api_key=_nvidia_key())

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        if self.backend == "hash":  # deterministic bag-of-words, for tests only
            vecs = np.zeros((len(texts), 512), dtype=np.float32)
            for r, t in enumerate(texts):
                for w in re.findall(r"[a-z0-9]+", t.lower()):
                    vecs[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
        elif self.backend == "local":
            prefix = "Represent this sentence for searching relevant passages: " if kind == "query" else ""
            vecs = self.model.encode([prefix + t for t in texts], batch_size=64, convert_to_numpy=True)
        else:
            vecs = []
            for i in range(0, len(texts), 50):
                resp = self.client.embeddings.create(
                    model=NV_EMBED_MODEL, input=texts[i:i + 50],
                    extra_body={"input_type": kind, "truncate": "END"})
                vecs.extend(d.embedding for d in resp.data)
            vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


def build_index(report_dir: Path, index_dir: Path = INDEX_DIR, backend: str = EMBED_BACKEND) -> int:
    chunks = load_reports(report_dir)
    if not chunks:
        raise SystemExit(f"No reports found under {report_dir}")
    vecs = Embedder(backend).embed([c.text for c in chunks], kind="passage")
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "vectors.npy", vecs)
    pd.DataFrame([asdict(c) for c in chunks]).to_json(index_dir / "chunks.jsonl", orient="records", lines=True)
    (index_dir / "meta.json").write_text(json.dumps({"backend": backend, "n_chunks": len(chunks)}))
    return len(chunks)


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.chunks = pd.read_json(index_dir / "chunks.jsonl", orient="records", lines=True)
        self.vecs = np.load(index_dir / "vectors.npy")
        self.embedder = Embedder(meta["backend"])

    def search(self, query: str, state: str | None = None, year: int | None = None, k: int = 5) -> list[dict]:
        mask = np.ones(len(self.chunks), dtype=bool)
        if state:  # filter BEFORE ranking so Iowa text never answers a Maryland question
            mask &= self.chunks["state"].isin([state, "US"]).to_numpy()
        if year:
            mask &= (self.chunks["year"] == year).to_numpy() | self.chunks["year"].isna().to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        q = self.embedder.embed([query], kind="query")[0]
        scores = self.vecs[idx] @ q
        order = idx[np.argsort(-scores)[:k]]
        out = []
        for i in order:
            row = self.chunks.iloc[i]
            out.append({"text": row.text, "source": row.source, "page": int(row.page),
                        "state": row.state, "year": None if pd.isna(row.year) else int(row.year),
                        "score": round(float(self.vecs[i] @ q), 3)})
        return out


_RETRIEVER: Retriever | None = None


def retrieve_reports(query: str, state: str | None = None, year: int | None = None, k: int = 5) -> list[dict]:
    """Tool: passages from state reports, each with source + page for citation."""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER.search(query, state, year, k)


# ---------------------------------------------------------------- 3. QuickStats (structured, no RAG)

CANONICAL_SERIES = {"CORN, GRAIN - ACRES HARVESTED", "CORN, SILAGE - ACRES HARVESTED", "CORN - ACRES PLANTED"}


def load_quickstats(path: Path = QUICKSTATS_CSV) -> pd.DataFrame:
    """Load the challenge QuickStats CSV and keep one clean series per year.

    The raw file mixes several rows for the same year and crop:
      - reference_period: final YEAR value plus AUG/SEP/OCT/NOV forecasts and JUN/MAR acreage
      - domain / short_description: ORGANIC and IRRIGATED subsets on top of the TOTAL
      - source: SURVEY and CENSUS rows for the same county in Census years
    Summing them all counted grain about 5x. We keep only final, total, all-practice values.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("state_name", "county_name", "aggregation_level", "crop_use", "measurement",
                "source", "domain", "reference_period", "short_description"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["acres"] = pd.to_numeric(df["value"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    df = df.dropna(subset=["acres", "year"])  # drops (D), (S), (Z) and other suppressed codes
    df["year"] = df["year"].astype(int)
    if "reference_period" in df.columns:
        df = df[df["reference_period"] == "YEAR"]
    if "domain" in df.columns:
        df = df[df["domain"] == "TOTAL"]
    if "short_description" in df.columns:
        df = df[df["short_description"].isin(CANONICAL_SERIES)]
    return df


_QS: pd.DataFrame | None = None


def _totals(src: pd.DataFrame) -> tuple[float, float, float]:
    def total(use, meas):
        return float(src[(src["crop_use"] == use) & (src["measurement"] == meas)]["acres"].sum())
    return total("SILAGE", "HARVESTED"), total("GRAIN", "HARVESTED"), total("ALL_CORN", "PLANTED")


def silage_share_history(state: str, county: str | None = None, years: list[int] | None = None) -> dict:
    """Tool: silage share of harvested corn acres by year, from NASS QuickStats.

    State level: SURVEY state rows; in Census years (county-only download) the SURVEY county rows,
    which include an OTHER COUNTIES bucket and so add up to the full state. County level: CENSUS
    county rows when present (Census years), else SURVEY. Years where silage is suppressed are skipped.
    """
    global _QS
    if _QS is None:
        _QS = load_quickstats()
    df = _QS[_QS["state_name"] == state.upper()]
    if county:
        df = df[(df["aggregation_level"] == "COUNTY") & (df["county_name"] == county.upper())]
    if years:
        df = df[df["year"].isin(years)]

    rows = []
    for year, g in df.groupby("year"):
        note = None
        if county:
            src = g[g["source"] == "CENSUS"]
            if src.empty or src[src["crop_use"] == "SILAGE"].empty:
                src, note = g[g["source"] == "SURVEY"], "survey county estimate"
            else:
                note = "census"
        else:
            src = g[(g["aggregation_level"] == "STATE") & (g["source"] == "SURVEY")]
            if src.empty:
                src = g[(g["aggregation_level"] == "COUNTY") & (g["source"] == "SURVEY")]
                note = "summed from survey county rows (includes OTHER COUNTIES)"
            if src.empty:
                src = g[(g["aggregation_level"] == "COUNTY") & (g["source"] == "CENSUS")]
                note = "summed from census counties; suppressed counties missing, may undercount"
        silage, grain, planted = _totals(src)
        if silage == 0 or grain == 0:  # suppressed or missing for this year
            continue
        rows.append({"year": int(year), "silage_acres": silage, "grain_acres": grain,
                     "planted_acres": planted or None,
                     "silage_share": round(silage / (silage + grain), 4), "note": note})

    # undercounted years stay out of the range when there are enough clean years
    clean = [r["silage_share"] for r in rows if not (r["note"] or "").startswith("summed from census")]
    shares = clean if len(clean) >= 3 else [r["silage_share"] for r in rows]
    summary = ({"n_years": len(shares), "min": min(shares), "max": max(shares),
                "mean": round(float(np.mean(shares)), 4), "std": round(float(np.std(shares)), 4)}
               if shares else {"n_years": 0})
    return {"state": state, "county": county, "by_year": rows, "summary": summary,
            "source": "USDA NASS Quick Stats, final annual SURVEY/CENSUS values, CORN GRAIN/SILAGE ACRES HARVESTED"}


def check_prediction(state: str, pred_silage_acres: float, pred_grain_acres: float,
                     planted_acres: float | None = None, county: str | None = None,
                     tolerance: float = 0.05) -> dict:
    """Tool: deterministic plausibility checks. The LLM explains these, it never overrides them."""
    hist = silage_share_history(state, county)
    s = hist["summary"]
    share = pred_silage_acres / max(pred_silage_acres + pred_grain_acres, 1)
    checks = []

    if s.get("n_years", 0) >= 3:
        lo, hi = s["min"] - tolerance, s["max"] + tolerance
        if s["min"] <= share <= s["max"]:
            status = "pass"
        elif lo <= share <= hi:
            status = "warn"
        else:
            status = "fail"
        checks.append({"check": "silage_share_in_history", "status": status,
                       "observed": round(share, 4), "expected_range": [s["min"], s["max"]],
                       "history_mean": s["mean"]})
    else:
        checks.append({"check": "silage_share_in_history", "status": "skip",
                       "reason": "fewer than 3 years of usable QuickStats history"})

    if planted_acres:
        harvested = pred_silage_acres + pred_grain_acres
        ratio = harvested / planted_acres
        checks.append({"check": "harvested_le_planted",
                       "status": "pass" if ratio <= 1.02 else "fail",
                       "observed_ratio": round(ratio, 3)})

    worst = "fail" if any(c["status"] == "fail" for c in checks) else (
        "warn" if any(c["status"] == "warn" for c in checks) else "pass")
    return {"state": state, "county": county, "overall": worst, "checks": checks}


# ---------------------------------------------------------------- 4. tool schemas + critic loop

TOOLS = [
    {"type": "function", "function": {
        "name": "retrieve_reports",
        "description": "Search NASS state reports for context such as usual harvest timing, weather events, "
                       "and notes on silage. Returns passages with source and page for citation.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "state": {"type": "string", "description": "Full state name, e.g. Maryland"},
            "year": {"type": "integer"},
            "k": {"type": "integer", "default": 5}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "silage_share_history",
        "description": "Historical silage share of harvested corn acres from NASS QuickStats, by year.",
        "parameters": {"type": "object", "properties": {
            "state": {"type": "string"}, "county": {"type": "string"},
            "years": {"type": "array", "items": {"type": "integer"}}},
            "required": ["state"]}}},
    {"type": "function", "function": {
        "name": "check_prediction",
        "description": "Deterministic plausibility checks of predicted silage and grain acres against history.",
        "parameters": {"type": "object", "properties": {
            "state": {"type": "string"}, "county": {"type": "string"},
            "pred_silage_acres": {"type": "number"}, "pred_grain_acres": {"type": "number"},
            "planted_acres": {"type": "number"}},
            "required": ["state", "pred_silage_acres", "pred_grain_acres"]}}},
]
TOOL_FUNCS = {"retrieve_reports": retrieve_reports,
              "silage_share_history": silage_share_history,
              "check_prediction": check_prediction}

CRITIC_PROMPT = """You are the critic in a USDA NASS pipeline that classifies corn fields as silage or grain.
You receive a prediction summary for one region. Your job is to decide whether it is plausible.
Rules:
- Always call check_prediction first. Its pass/warn/fail is final: you explain it, you never override it.
- Use retrieve_reports to find context (usual silage harvest window, weather events, dairy notes) and cite source + page.
- Never invent numbers. Every number you mention must come from a tool result or the input.
- Finish with ONLY a JSON object:
{"overall": "pass|warn|fail", "flags": [{"check": str, "status": str, "why": str,
  "evidence": [{"source": str, "page": int, "quote": str}]}],
 "suggested_action": "accept | rerun_with_radar | rerun_with_regional_threshold | abstain_region",
 "summary": str}"""


def _nvidia_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    candidates = [RAG_DIR / ".env", REPO_DIR / ".env",
                  Path(os.environ.get("KEEBOH_WORKSPACE_DIR", "/workspace")) / ".env"]
    for env in candidates:
        if key:
            break
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip() == "NVIDIA_API_KEY":
                    key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("Set NVIDIA_API_KEY in FieldVision/.env (never commit it)")
    return key


def run_critic(prediction: dict, max_steps: int = 6, model: str = CHAT_MODEL) -> dict:
    """prediction example: {"state": "Maryland", "year": 2022, "pred_silage_acres": 61000,
    "pred_grain_acres": 380000, "median_cut_date": "2022-09-02", "n_fields": 4123, "abstain_rate": 0.06}"""
    from openai import OpenAI
    client = OpenAI(base_url=NV_BASE_URL, api_key=_nvidia_key())
    messages = [{"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": json.dumps(prediction)}]
    trace = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=messages, tools=TOOLS,
                                              tool_choice="auto", temperature=0, max_tokens=1500)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            match = re.search(r"\{.*\}", text, flags=re.S)
            result = json.loads(match.group(0)) if match else {"overall": "warn", "summary": text}
            result["trace"] = trace
            return result
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            try:
                output = TOOL_FUNCS[call.function.name](**args)
            except Exception as exc:  # report tool errors back to the model instead of crashing
                output = {"error": f"{type(exc).__name__}: {exc}"}
            trace.append({"tool": call.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(output, default=str)[:12000]})
    return {"overall": "warn", "summary": "critic hit max_steps", "trace": trace}


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--reports", default=str(DATASET_DIR)); b.add_argument("--index", default=str(INDEX_DIR))
    b.add_argument("--backend", default=EMBED_BACKEND)
    a = sub.add_parser("ask"); a.add_argument("query"); a.add_argument("--state"); a.add_argument("--year", type=int)
    a.add_argument("--k", type=int, default=5)
    s = sub.add_parser("share"); s.add_argument("--state", required=True); s.add_argument("--county")
    c = sub.add_parser("check"); c.add_argument("--state", required=True); c.add_argument("--county")
    c.add_argument("--silage", type=float, required=True); c.add_argument("--grain", type=float, required=True)
    c.add_argument("--planted", type=float)
    r = sub.add_parser("critic"); r.add_argument("prediction_json")
    args = ap.parse_args()

    if args.cmd == "build":
        n = build_index(Path(args.reports), Path(args.index), args.backend)
        print(f"Indexed {n} chunks into {args.index}")
    elif args.cmd == "ask":
        for hit in retrieve_reports(args.query, args.state, args.year, args.k):
            print(f"[{hit['score']}] {hit['source']} p{hit['page']} ({hit['state']}, {hit['year']})")
            print("   " + hit["text"][:300].replace("\n", " ") + "\n")
    elif args.cmd == "share":
        print(json.dumps(silage_share_history(args.state, args.county), indent=2))
    elif args.cmd == "check":
        print(json.dumps(check_prediction(args.state, args.silage, args.grain, args.planted, args.county), indent=2))
    elif args.cmd == "critic":
        print(json.dumps(run_critic(json.loads(Path(args.prediction_json).read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
