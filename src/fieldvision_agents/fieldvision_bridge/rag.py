"""The team's RAG pipeline (FieldVision/rag-pipeline) set up for the agents.

Two pieces are used:
  - the critic: code checks against Quick Stats history decide pass/warn/fail and the suggested
    action; Nemotron Ultra only explains the verdict, citing NASS report passages;
  - report search: passages from NASS Crop Progress reports and handbooks, with source and page,
    for analyst questions about usual harvest timing and the season.
Numbers in a verdict come from Quick Stats, never from the vector index.

The index was embedded with sentence-transformers (BAAI/bge-small-en-v1.5). Embedding a question needs
that package and torch, about a gigabyte; without them, load_rag installs KeywordRetriever, which ranks
the same indexed passages by shared words, with the same state and year filters.
"""

import importlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter

from .config import COUNTY_DATA, RAG_PIPELINE

STATE_FIPS = {"24": "Maryland"}       # filled out from the critic module when it loads
_WORD = re.compile(r"[a-z]{3,}")


class KeywordRetriever:
    """BM25 over the index's chunks.jsonl: the rag_pipeline.Retriever interface without embeddings."""

    def __init__(self, index_dir, k1=1.5, b=0.75):
        self.chunks = [json.loads(line) for line in (index_dir / "chunks.jsonl").read_text().splitlines()
                       if line.strip()]
        self.terms = [Counter(_WORD.findall(c["text"].lower())) for c in self.chunks]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(w for t in self.terms for w in t)
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, state=None, year=None, k=5):
        words = set(_WORD.findall(query.lower()))
        scored = []
        for c, t, n in zip(self.chunks, self.terms, self.lengths):
            if state and c.get("state") not in (state, "US"):
                continue
            if year and c.get("year") not in (year, None):
                continue
            s = sum(self.idf.get(w, 0) * t[w] * (self.k1 + 1) / (t[w] + self.k1 * (1 - self.b + self.b * n / self.avg))
                    for w in words if w in t)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [{"text": c["text"], "source": c["source"], "page": c.get("page"), "state": c.get("state"),
                 "year": c.get("year"), "score": round(s, 3), "retriever": "keyword"} for s, c in scored[:k]]


def embeddings_available():
    return importlib.util.find_spec("sentence_transformers") is not None


def load_rag(rag_dir=RAG_PIPELINE):
    """Import rag_pipeline and critic_toolbox from the team's folder."""
    for p in (str(rag_dir),):
        if p not in sys.path:
            sys.path.insert(0, p)
    critic = importlib.import_module("critic_toolbox")
    STATE_FIPS.update(critic.STATE_FIPS)
    rag = importlib.import_module("rag_pipeline")
    index_dir = rag.INDEX_DIR
    meta = json.loads((index_dir / "meta.json").read_text()) if (index_dir / "meta.json").is_file() else {}
    if rag._RETRIEVER is None and meta.get("backend") == "local" and not embeddings_available():
        rag._RETRIEVER = KeywordRetriever(index_dir)
    return rag, critic


class _CountyResult:
    """What CriticToolbox reads from a toolbox: aggregate() and data_root."""

    def __init__(self, data_root):
        self.data_root, self.result = data_root, None

    def aggregate(self, fips, year):
        return self.result


def make_rag_critic(use_llm=True, critic_module=None, county_data=COUNTY_DATA):
    """A critic for FieldVisionToolbox(critic=...): critic(fips, year, county_result, policy)."""
    module = critic_module or load_rag()[1]
    shim = _CountyResult(county_data)
    tb = module.CriticToolbox(shim, use_llm=use_llm)

    def critic(fips, year, county_result, policy="standard"):
        key = (str(fips), int(year))
        shim.result = county_result
        tb._agg.pop(key, None)
        tb._policy[key] = policy
        return tb.run_critic(fips, year)

    return critic


def make_report_search(rag_module=None, max_chars=600):
    """search(query, fips, year=None, k=3) -> NASS report passages with source and page."""
    module = rag_module or load_rag()[0]

    def search(query, fips, year=None, k=3):
        state = STATE_FIPS.get(str(fips).zfill(5)[:2])
        hits = module.retrieve_reports(query, state=state, year=year, k=max(1, min(int(k), 5)))
        return [{"source": h.get("source"), "page": h.get("page"), "state": h.get("state"),
                 "year": h.get("year"), "text": (h.get("text") or "")[:max_chars]} for h in hits]

    return search
