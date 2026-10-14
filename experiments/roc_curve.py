"""ROC curves on the held-out test set, without re-running the full train.py.

Why this script exists
- train.py tunes four models (RandomizedSearchCV) and keeps only the chosen
  one in the bundle. Re-running it just for one chart takes a long time.
- The tuned settings of all four models are already saved in
  reports/metrics.json ("best_params"). Refitting each pipeline with those
  settings on the same train split (same RANDOM_STATE) rebuilds exactly the
  models that RandomizedSearchCV refit, so the curves match train.py.

What the chart shows
- One ROC curve per candidate model (uncalibrated, same as the PR-curve chart),
  with test ROC-AUC in the legend.
- The deployed operating point: the calibrated chosen model at the
  cost-optimal threshold, read from metrics.json["test_evaluation"].
- The diagonal = a random model (AUC 0.5).

Note: ROC-AUC looks flattering when only ~16% are leavers, which is why the
project selects models on PR-AUC. Show this chart next to pr_curves_test.png.

Run from the repo root:  python -m experiments.roc_curve
Output: reports/figures/roc_curves_test.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src import config, data, model  # noqa: E402


def _restore(params: dict) -> dict:
    # JSON turns tuples into lists; MLP hidden_layer_sizes must be a tuple again
    return {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}


def main():
    metrics = json.loads(config.METRICS_PATH.read_text())
    df = data.prepare(data.load_raw())
    lists = data.feature_lists()
    train, test = data.split(df)
    X_train, y_train = data.X_y(train, lists)
    X_test, y_test = data.X_y(test, lists)

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    specs = model.model_specs(pos_weight)
    best_name = metrics["best_model"]

    plt.figure(figsize=(5.5, 4.6))
    aucs = {}
    for name, spec in specs.items():
        pipe = model.build_pipeline(spec["estimator"], lists, spec["smote"])
        best = _restore(metrics["model_comparison"][name]["best_params"])
        pipe.set_params(**{f"clf__{k}": v for k, v in best.items()})
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        aucs[name] = float(roc_auc_score(y_test, p))
        chosen = name == best_name
        plt.plot(fpr, tpr, lw=2.2 if chosen else 1.2,
                 label=f"{name}{' (chosen)' if chosen else ''}  AUC {aucs[name]:.3f}")

    ev = metrics["test_evaluation"]
    cm = ev["confusion_matrix"]
    op_fpr = cm["fp"] / (cm["fp"] + cm["tn"])
    op_tpr = cm["tp"] / (cm["tp"] + cm["fn"])
    plt.scatter([op_fpr], [op_tpr], s=60, c="k", zorder=5,
                label=f"Deployed: threshold {ev['threshold']:.2f} "
                      f"(TPR {op_tpr:.2f}, FPR {op_fpr:.2f})")
    plt.plot([0, 1], [0, 1], ls=":", c="grey", label="Random  AUC 0.500")
    plt.xlim(0, 1); plt.ylim(0, 1.01)
    plt.xlabel("False positive rate (stayers wrongly flagged)")
    plt.ylabel("True positive rate (leavers caught)")
    plt.title("ROC curves (test)")
    plt.legend(fontsize=7, loc="lower right")
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    out = config.FIG_DIR / "roc_curves_test.png"
    plt.savefig(out, dpi=130)
    plt.close()

    for name, a in aucs.items():
        print(f"{name:<24} test ROC-AUC {a:.3f}")
    print(f"Deployed point: TPR {op_tpr:.3f}, FPR {op_fpr:.3f} at threshold {ev['threshold']:.2f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
