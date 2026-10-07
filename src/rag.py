"""Track 5 - Enterprise GenAI / NLP: the HR Copilot (RAG).

    handbook.md -> chunks -> embeddings -> vector index (FAISS)
    question    -> embedding -> top-k chunks (cosine or Euclidean) -> LLM prompt
    LLM         -> retention brief grounded in the retrieved clauses + SHAP drivers

Embeddings: sentence-transformers 'all-MiniLM-L6-v2' (Hugging Face). If it is
not installed or cannot download, falls back to TF-IDF so the demo never breaks.
LLM: Gemini (Google AI Studio) or Claude via API key in the environment; with
no key, a template brief is returned instead.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from src import config

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def load_chunks(path=None, max_words: int = 120) -> pd.DataFrame:
    path = path or config.HANDBOOK
    """Split the handbook on '## ' headings, then long sections into paragraphs."""
    text = path.read_text(encoding="utf-8")
    rows = []
    for block in re.split(r"\n(?=## )", text):
        if not block.startswith("## "):
            continue
        heading, _, body = block.partition("\n")
        section = heading.lstrip("# ").strip()
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        buf = []
        for p in paras:
            if buf and len(" ".join(buf + [p]).split()) > max_words:
                rows.append({"section": section, "text": " ".join(buf)})
                buf = []
            buf.append(p)
        if buf:
            rows.append({"section": section, "text": " ".join(buf)})
    df = pd.DataFrame(rows)
    df["chunk_id"] = range(len(df))
    return df


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
class Embedder:
    def __init__(self, prefer_transformer: bool = True):
        self.backend = "tfidf"
        if prefer_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(EMBED_MODEL)
                self.backend = "sentence-transformers"
            except Exception:  # not installed, or no internet to download weights
                pass

    def fit(self, texts: list[str]):
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return np.asarray(self._st.encode(texts), dtype="float32")
        return self._tfidf.transform(texts).toarray().astype("float32")


# --------------------------------------------------------------------------
# Vector index
# --------------------------------------------------------------------------
class VectorIndex:
    """Two indexes over the same chunks so we can compare distance metrics.
    cosine    = inner product on L2-normalised vectors (direction only)
    euclidean = L2 distance on raw vectors (direction AND length)"""

    def __init__(self, vectors: np.ndarray):
        self.raw = vectors.astype("float32")
        self.norm = self.raw / np.clip(np.linalg.norm(self.raw, axis=1, keepdims=True), 1e-12, None)
        try:
            import faiss
            self.ip = faiss.IndexFlatIP(self.norm.shape[1]); self.ip.add(self.norm)
            self.l2 = faiss.IndexFlatL2(self.raw.shape[1]); self.l2.add(self.raw)
            self.backend = "faiss"
        except ImportError:
            self.backend = "numpy"

    def search(self, q: np.ndarray, k: int = 3, metric: str = "cosine"):
        q = q.astype("float32").reshape(1, -1)
        if metric == "cosine":
            qn = q / max(np.linalg.norm(q), 1e-12)
            if self.backend == "faiss":
                s, i = self.ip.search(qn, k)
                return i[0], s[0]
            s = self.norm @ qn[0]
            i = np.argsort(-s)[:k]
            return i, s[i]
        if self.backend == "faiss":
            d, i = self.l2.search(q, k)
            return i[0], np.sqrt(d[0])
        d = np.linalg.norm(self.raw - q, axis=1)
        i = np.argsort(d)[:k]
        return i, d[i]


class Retriever:
    def __init__(self, chunks: pd.DataFrame, prefer_transformer: bool = True):
        self.chunks = chunks
        self.embedder = Embedder(prefer_transformer).fit(chunks.text.tolist())
        self.index = VectorIndex(self.embedder.encode(chunks.text.tolist()))

    def retrieve(self, query: str, k: int = 3, metric: str = "cosine") -> pd.DataFrame:
        idx, scores = self.index.search(self.embedder.encode([query])[0], k, metric)
        out = self.chunks.iloc[idx].copy()
        out["score"] = scores
        out["metric"] = metric
        return out


def evaluate_retrieval(retriever: Retriever, eval_df: pd.DataFrame, k: int = 3) -> dict:
    """eval_df columns: question, relevant_section (hand-labelled by the team).
    hit@k = right section appears in the top k; MRR = 1 / rank of first hit."""
    out = {}
    for metric in ["cosine", "euclidean"]:
        hits, rr = [], []
        for row in eval_df.itertuples():
            secs = retriever.retrieve(row.question, k, metric).section.tolist()
            hit = row.relevant_section in secs
            hits.append(hit)
            rr.append(1 / (secs.index(row.relevant_section) + 1) if hit else 0)
        out[metric] = {f"hit_at_{k}": float(np.mean(hits)), "mrr": float(np.mean(rr))}
    out["n_questions"] = int(len(eval_df))
    out["embedding_backend"] = retriever.embedder.backend
    return out


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def build_prompt(question: str, employee_summary: str, drivers: pd.DataFrame,
                 recommendations: pd.DataFrame, retrieved: pd.DataFrame) -> str:
    clauses = "\n".join(f"[{r.section}] {r.text}" for r in retrieved.itertuples())
    drv = "\n".join(f"- {r.feature} = {r.value}: {r.direction} (SHAP {r.shap:+.3f})" for r in drivers.itertuples())
    recs = "\n".join(f"- {r.name}" for r in recommendations.itertuples())
    return f"""You are an HR business partner assistant. Write a short retention brief
for a line manager (max 150 words, plain language).

Rules:
- Use ONLY the policy clauses below. Cite the section name in [brackets] for every policy you mention.
- If the clauses do not cover something, say so. Do not invent policies, amounts or eligibility.
- The risk drivers are statistical associations from a model, not proven causes. Say "linked to", not "caused by".

Manager's question: {question}

Employee: {employee_summary}

Model risk drivers:
{drv}

Recommended interventions (from the recommender):
{recs}

Policy clauses:
{clauses}
"""


def generate(prompt: str) -> tuple[str, str]:
    """Returns (text, provider). Provider chosen by environment variables:
    LLM_PROVIDER = gemini | anthropic ; GEMINI_API_KEY / ANTHROPIC_API_KEY ;
    LLM_MODEL = model name (check the provider's docs for current names)."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    try:
        if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
            from google import genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.generate_content(model=os.getenv("LLM_MODEL", "gemini-2.5-flash"), contents=prompt)
            return resp.text, "gemini"
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text, "anthropic"
    except Exception as e:  # network or quota problems must not break a live demo
        return f"(LLM call failed: {e})\n\n" + _template_brief(prompt), "template"
    return _template_brief(prompt), "template"


def _template_brief(prompt: str) -> str:
    """No-LLM fallback: returns the grounded facts without generated prose."""
    body = prompt.split("Employee:", 1)[1]
    return "No LLM configured - showing the grounded inputs the LLM would use:\n\nEmployee:" + body
