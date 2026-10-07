"""Phase 3, Track 5: which retrieval set-up finds the right policy?

Compared on data/knowledge_base/retrieval_eval.csv (33 hand-labelled manager
questions, written before any retrieval run, worded differently from the
handbook on purpose):
  * TF-IDF, L2-normalised (sklearn default)  cosine vs Euclidean
  * TF-IDF, NOT normalised                   cosine vs Euclidean
  * sentence-transformers all-MiniLM-L6-v2   cosine vs Euclidean
    (only where the model can be downloaded; the build sandbox cannot reach
    Hugging Face, so run this script on a laptop for the full table)

Expectation stated in docs/PHASE3_PLAN.md: on unit-length vectors cosine and
Euclidean give the same ranking (|a-b|^2 = 2 - 2cos), so they can only differ
on un-normalised vectors.

Decision rule: the app uses the set-up with the best MRR; ties go to the one
with fewer dependencies (TF-IDF).

Run from the repo root:  python -m experiments.phase3_rag
"""
import json

import numpy as np
import pandas as pd

from src import config, rag


def main():
    chunks = rag.load_chunks()
    ev = pd.read_csv(config.RAG_EVAL)
    out, rows = {"n_questions": len(ev), "n_chunks": len(chunks)}, []
    setups = {"tfidf_l2": dict(prefer_transformer=False, tfidf_norm="l2"),
              "tfidf_raw": dict(prefer_transformer=False, tfidf_norm=None),
              "minilm": dict(prefer_transformer=True)}
    for name, kw in setups.items():
        r = rag.Retriever(chunks, **kw)
        if name == "minilm" and r.embedder.backend != "sentence-transformers":
            out[name] = {"status": "skipped - sentence-transformers not installed or model not downloadable here"}
            continue
        res = rag.evaluate_retrieval(r, ev)
        norms = np.linalg.norm(r.index.raw, axis=1)
        res["vector_norm_range"] = [float(norms.min()), float(norms.max())]
        out[name] = res
        for m in ["cosine", "euclidean"]:
            rows.append({"setup": name, "metric": m, **{k: res[m][k] for k in ["hit_at_1", "hit_at_3", "mrr"]}})
    table = pd.DataFrame(rows)
    best = table.sort_values(["mrr", "setup"], ascending=[False, True]).iloc[0]
    tf_best = table[table.setup.str.startswith("tfidf")].mrr.max()
    out["table"] = table.round(4).to_dict("records")
    out["decision"] = (f"{best.setup}/{best.metric}" if best.mrr > tf_best + 1e-9
                       else "tfidf_l2/cosine (best or tied; fewest dependencies)")
    (config.REPORTS_DIR / "phase3_rag.json").write_text(json.dumps(out, indent=2))
    print(table.round(3).to_string(index=False))
    print("decision:", out["decision"])


if __name__ == "__main__":
    main()
