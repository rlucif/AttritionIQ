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
    """tfidf_norm='l2' (sklearn default) makes every vector unit length, so
    cosine and Euclidean rank identically (|a-b|^2 = 2 - 2cos). Phase 3 uses
    tfidf_norm=None to show when the two metrics actually differ."""

    def __init__(self, prefer_transformer: bool = True, tfidf_norm: str | None = "l2"):
        self.backend = "tfidf"
        self.tfidf_norm = tfidf_norm
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
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                                          norm=self.tfidf_norm).fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return np.asarray(self._st.encode(texts), dtype="float32")
        return self._tfidf.transform(texts).toarray().astype("float32")


# --------------------------------------------------------------------------
# Vector index
# --------------------------------------------------------------------------
TIE_TOL = 1e-6  # see VectorIndex.search


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
        """Top-k chunk indices and scores (cosine: similarity, higher = better;
        euclidean: distance, lower = better).

        faiss (when installed) picks the candidates; they are then re-scored
        exactly in float64 and near-ties (within 1e-6) broken by handbook order, so results are
        the same on every machine. Phase 3 fix: faiss's float32 scores put
        equal-score chunks in arbitrary order, so on a laptop with faiss,
        TF-IDF cosine and Euclidean differed slightly (MRR 0.842 vs 0.844)."""
        q = q.astype("float32").reshape(1, -1)
        n = len(self.raw)
        if self.backend == "faiss":
            pool = min(n, max(5 * k, 50))  # candidate pool, then exact re-scoring
            if metric == "cosine":
                qn = q / max(np.linalg.norm(q), 1e-12)
                cand = self.ip.search(qn, pool)[1][0]
            else:
                cand = self.l2.search(q, pool)[1][0]
            cand = cand[cand >= 0]
        else:
            cand = np.arange(n)
        X, qq = self.raw[cand].astype("float64"), q[0].astype("float64")
        if metric == "cosine":
            score = (X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)) @ (
                qq / max(np.linalg.norm(qq), 1e-12))
            key = -score
        else:
            score = np.linalg.norm(X - qq, axis=1)
            key = score
        # Scores within 1e-6 of each other count as tied (vectors are stored in
        # float32, whose rounding noise is ~1e-7); tied chunks keep handbook order.
        srt = np.argsort(key, kind="stable")
        group = np.r_[0, np.cumsum(np.diff(key[srt]) > TIE_TOL)]
        tie_group = np.empty(len(cand), dtype=int); tie_group[srt] = group
        order = np.lexsort((cand, tie_group))[:k]
        return cand[order], score[order]


class Retriever:
    def __init__(self, chunks: pd.DataFrame, prefer_transformer: bool = True, tfidf_norm: str | None = "l2"):
        self.chunks = chunks
        self.embedder = Embedder(prefer_transformer, tfidf_norm).fit(chunks.text.tolist())
        self.index = VectorIndex(self.embedder.encode(chunks.text.tolist()))

    def retrieve(self, query: str, k: int = 3, metric: str = "cosine") -> pd.DataFrame:
        """Top-k chunks. `matched` is False when the query vector is all zeros
        (TF-IDF: no word in common with the handbook): the ranking is then
        meaningless, and the app should say "no relevant policy found"."""
        q = self.embedder.encode([query])[0]
        idx, scores = self.index.search(q, k, metric)
        out = self.chunks.iloc[idx].copy()
        out["score"] = scores
        out["metric"] = metric
        out["matched"] = bool(np.linalg.norm(q) > 0)
        return out


def evaluate_retrieval(retriever: Retriever, eval_df: pd.DataFrame, k: int = 3) -> dict:
    """eval_df columns: question, relevant_section (hand-labelled by the team).
    hit@1 / hit@k = right section in the top 1 / top k chunks;
    MRR = mean of 1 / rank of the first chunk from the right section."""
    out = {}
    n = len(retriever.chunks)
    for metric in ["cosine", "euclidean"]:
        h1, hk, rr, misses = [], [], [], []
        for row in eval_df.itertuples():
            res = retriever.retrieve(row.question, n, metric)
            secs = res.section.tolist()
            # an all-zero query vector retrieves nothing meaningful: count it as a miss
            rank = (secs.index(row.relevant_section) + 1
                    if res.matched.iloc[0] and row.relevant_section in secs else None)
            h1.append(rank == 1)
            hk.append(rank is not None and rank <= k)
            rr.append(1 / rank if rank else 0)
            if rank != 1:
                misses.append({"question": row.question, "expected": row.relevant_section,
                               "got": secs[0] if res.matched.iloc[0] else "(no word in common)",
                               "rank": rank})
        out[metric] = {"hit_at_1": float(np.mean(h1)), f"hit_at_{k}": float(np.mean(hk)),
                       "mrr": float(np.mean(rr)), "not_first": misses}
    out["n_questions"] = int(len(eval_df))
    out["n_chunks"] = int(n)
    out["embedding_backend"] = retriever.embedder.backend
    return out


# Plain words for model features, so a retrieval query built from SHAP drivers
# uses the handbook's vocabulary ("MonthlyIncome" matches nothing; "salary pay" does).
FEATURE_WORDS = {
    "OverTime": "overtime workload hours", "WorkLifeBalance": "workload flexible working",
    "MonthlyIncome": "salary pay compensation", "PayGapPct": "pay equity salary range",
    "PercentSalaryHike": "salary adjustment pay", "StockOptionLevel": "stock equity retention grant",
    "JobLevel": "promotion level", "YearsSinceLastPromotion": "promotion",
    "YearsInCurrentRole": "role rotation internal mobility", "TotalWorkingYears": "career progression",
    "YearsAtCompany": "tenure stay interview", "YearsWithCurrManager": "manager team",
    "RelationshipSatisfaction": "manager team relationship", "EnvironmentSatisfaction": "working environment team",
    "JobSatisfaction": "stay interview role", "JobInvolvement": "development mentoring",
    "TrainingTimesLastYear": "training learning development", "DistanceFromHome": "commute remote work",
    "BusinessTravel": "business travel", "NumCompaniesWorked": "stay interview retention",
    "JobRole": "role rotation internal mobility", "Department": "internal mobility team",
}


def driver_query(question: str, features) -> str:
    """Question + plain-word versions of the employee's top risk drivers."""
    return " ".join([question] + [FEATURE_WORDS.get(f, f) for f in features])


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
