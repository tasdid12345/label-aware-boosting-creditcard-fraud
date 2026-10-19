"""Label-free Isolation Forest (baseline) vs class-balanced, CV-tuned LightGBM (improved)
on the OpenML credit-card fraud data, evaluated on a later time-ordered test period."""
import io, json, math, sys, time
import numpy as np
import pandas as pd
import requests
from scipy.io import arff
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
SEEDS = [42, 43, 44]          # both models are randomized -> run every seed
TEST_FRAC = 0.30              # time-ordered: the last 30% of transactions is the test period
META_URL = "https://openml.org/api/v1/json/data/1597"
SOURCE_URL = "https://www.openml.org/d/1597"


def get(url, tries=5):
    """GET with timeout=60 and exponential backoff on HTTP 429/5xx and network errors."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"  download failed ({e}); retrying in {2 ** i}s")
            time.sleep(2 ** i)


def load_data():
    """Download OpenML dataset 1597 (ARFF) and return a DataFrame with a 0/1 'Class' column."""
    desc = get(META_URL).json()["data_set_description"]
    print("Downloading", desc["name"], "from", desc["url"])
    text = get(desc["url"]).content.decode("utf-8", errors="replace")
    # Newer SciPy parses ARFF headers with str regexes, so pass a text stream, not bytes.
    data, _ = arff.loadarff(io.StringIO(text))
    df = pd.DataFrame(data)
    df["Class"] = df["Class"].apply(lambda v: int(v.decode()) if isinstance(v, bytes) else int(v))
    return df


def metrics_at_alert(y, s, rate=0.01):
    """Recall and F1 when the top `rate` fraction of test transactions (by score) is flagged."""
    k = max(1, int(math.ceil(rate * len(y))))
    top = np.argsort(-s)[:k]
    tp = y[top].sum()
    prec, rec = tp / k, tp / y.sum()
    f1 = 2 * prec * rec / (prec + rec) if tp > 0 else 0.0
    return rec, f1


def evaluate(y, s):
    """Shared evaluation for both models: PR-AUC (main), ROC-AUC, recall/F1 at a 1% alert rate."""
    rec, f1 = metrics_at_alert(y, s)
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "recall_1pct": rec, "f1_1pct": f1}


def agg(runs, key):
    """Mean and population std across seeds."""
    vals = [r[key] for r in runs]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    df = load_data().sort_values("Time", kind="mergesort").reset_index(drop=True)
    feats = [c for c in df.columns if c != "Class"]
    X, y = df[feats].to_numpy(float), df["Class"].to_numpy(int)
    n = len(df)
    ntr = int(round((1 - TEST_FRAC) * n))           # earlier 70% -> train, later 30% -> test
    Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], y[:ntr], y[ntr:]
    if ytr.sum() == 0 or yte.sum() == 0:
        sys.exit("ERROR: a split has no fraud labels; cannot compute meaningful metrics.")
    print(f"Rows={n} train={ntr} test={n - ntr} train_fraud={ytr.sum()} test_fraud={yte.sum()}")

    # ---- Improved model: small CV search on the TRAINING period only (test set untouched) ----
    space = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1],
             "n_estimators": [200, 400], "min_child_samples": [20, 50, 100],
             "reg_lambda": [0.0, 1.0, 5.0]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_params, best_cv = None, -1.0
    for i, p in enumerate(ParameterSampler(space, n_iter=8, random_state=SEED)):
        fold_scores = []
        for tr, va in cv.split(Xtr, ytr):
            m = lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1,
                                   verbose=-1, **p).fit(Xtr[tr], ytr[tr])
            fold_scores.append(average_precision_score(ytr[va], m.predict_proba(Xtr[va])[:, 1]))
        cv_mean = float(np.mean(fold_scores))
        print(f"  CV {i + 1}/8 PR-AUC={cv_mean:.4f} {p}")
        if cv_mean > best_cv:
            best_cv, best_params = cv_mean, p
    print("Best CV params:", best_params)

    # ---- Final fit on full training period, single evaluation on the test period ----
    base_runs, imp_runs = [], []
    for s in SEEDS:
        iso = IsolationForest(n_estimators=200, random_state=s, n_jobs=-1).fit(Xtr)  # label-free
        base_runs.append(evaluate(yte, -iso.score_samples(Xte)))                    # high = anomalous
        clf = lgb.LGBMClassifier(class_weight="balanced", random_state=s, n_jobs=-1,
                                 verbose=-1, **best_params).fit(Xtr, ytr)
        imp_runs.append(evaluate(yte, clf.predict_proba(Xte)[:, 1]))
        print(f"seed {s}: baseline PR-AUC={base_runs[-1]['pr_auc']:.4f} "
              f"improved PR-AUC={imp_runs[-1]['pr_auc']:.4f}")

    b_m, b_s = agg(base_runs, "pr_auc")
    i_m, i_s = agg(imp_runs, "pr_auc")
    if not all(np.isfinite([b_m, i_m])):
        sys.exit("ERROR: non-finite metrics.")
    metrics = {
        "baseline_score": b_m, "improved_score": i_m, "improvement": i_m - b_m,
        "baseline_score_std": b_s, "improved_score_std": i_s,
        "baseline_roc_auc": agg(base_runs, "roc_auc")[0], "improved_roc_auc": agg(imp_runs, "roc_auc")[0],
        "baseline_recall_1pct": agg(base_runs, "recall_1pct")[0],
        "improved_recall_1pct": agg(imp_runs, "recall_1pct")[0],
        "baseline_f1": agg(base_runs, "f1_1pct")[0], "improved_f1": agg(imp_runs, "f1_1pct")[0],
        "improved_cv_pr_auc": best_cv, "test_fraud_count": int(yte.sum()),
    }
    results = {
        "question": "Can a class-balanced, CV-tuned LightGBM that uses the few confirmed fraud labels "
                    "beat a label-free unsupervised anomaly detector on time-ordered credit-card fraud detection?",
        "method": "Time-sorted 70/30 split; Isolation Forest (label-free) vs class-balanced LightGBM with "
                  "3-fold stratified CV random search on the training period; PR-AUC on the later test period; "
                  "seeds 42-44.",
        "dataset": {"name": "OpenML 1597 Credit Card Fraud", "source_url": SOURCE_URL, "n_rows": int(n),
                    "n_train": int(ntr), "n_test": int(n - ntr), "same_as_paper": False,
                    "baseline_is_papers_method": False},
        "metric_name": "PR-AUC (average precision) on test period",
        "higher_is_better": True,
        "baseline": "Isolation Forest (sklearn, 200 trees), label-free, fitted on training features only.",
        "improved": "LightGBM with class_weight='balanced', hyperparameters chosen by 3-fold stratified CV "
                    "random search (8 candidates) on training data using training fraud labels.",
        "metrics": metrics,
        "notes": f"Best CV params: {best_params}. Single time-ordered split; test fraud count is small so "
                 "estimates are noisy. CV is stratified, not time-aware. The Isolation Forest is a generic "
                 "label-free stand-in, not the paper's GraphGuard method and not a re-implementation of it. "
                 "Secondary F1 and recall use a top-1% alert rate. std is population std over 3 seeds. "
                 "Full dataset used (no subsampling) since runtime is well within budget. The features are "
                 "PCA components, so no graph structure is used.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")

    # ---- Optional chart: PR-AUC baseline vs improved ----
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Isolation Forest", "LightGBM (balanced)"], [b_m, i_m], yerr=[b_s, i_s], capsize=6,
           color=["#999999", "#2a7ab0"])
    ax.set_ylabel("PR-AUC on test period (higher is better)")
    ax.set_title("Fraud detection: baseline vs improved")
    fig.tight_layout()
    fig.savefig("figures/pr_auc_comparison.png", dpi=120)
    print("Saved figures/pr_auc_comparison.png")


if __name__ == "__main__":
    main()
