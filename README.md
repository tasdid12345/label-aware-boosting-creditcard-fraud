# Label-aware boosting vs. Isolation Forest on credit-card fraud

Question: can a class-balanced, CV-tuned LightGBM that uses the confirmed fraud labels beat a label-free Isolation Forest on time-ordered credit-card fraud detection?

Data: OpenML 1597 (Credit Card Fraud), sorted by `Time`. The first 70% is training data and the last 30% is the test period.

Method:
- Baseline: Isolation Forest (200 trees), fitted on training features only.
- Improved: LightGBM with `class_weight="balanced"`. Hyperparameters come from an 8-candidate random search scored by 3-fold stratified CV on the training period only.
- Metric: PR-AUC (average precision) on the test period, averaged over seeds 42, 43, 44. Secondary metrics are ROC-AUC, recall and F1 at a top-1% alert rate.

Run:
```
pip install -r requirements.txt
python analysis.py
```
Outputs: `results.json` and `figures/pr_auc_comparison.png`.

Fix note: the ARFF loader now receives decoded text instead of bytes, which newer SciPy versions require. The method, split, baseline and test set are unchanged.

## Results (latest run)

On the OpenML 1597 credit-card fraud dataset, a standard label-free Isolation Forest reached PR-AUC 0.038 on the later test period, while a class-balanced, CV-tuned LightGBM trained on training-period fraud labels reached 0.799 on the same test set (108 fraud cases, so noisy).

| Metric | Value |
|---|---|
| baseline_score | 0.03798 |
| improved_score | 0.7993 |
| improvement | 0.7613 |
| baseline_score_std | 0.005351 |
| improved_score_std | 0 |
| baseline_roc_auc | 0.9402 |
| improved_roc_auc | 0.9848 |
| baseline_recall_1pct | 0.4506 |
| improved_recall_1pct | 0.8611 |
| baseline_f1 | 0.1011 |
| improved_f1 | 0.1931 |
| improved_cv_pr_auc | 0.8658 |
| test_fraud_count | 108 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Best CV params: {'reg_lambda': 1.0, 'num_leaves': 15, 'n_estimators': 200, 'min_child_samples': 20, 'learning_rate': 0.1}. Single time-ordered split; test fraud count is small so estimates are noisy. CV is stratified, not time-aware. The Isolation Forest is a generic label-free stand-in, not the paper's GraphGuard method and not a re-implementation of it. Secondary F1 and recall use a top-1% alert rate. std is population std over 3 seeds. Full dataset used (no subsampling) since runtime is well within budget. The features are PCA components, so no graph structure is used.

## Reference

This project is motivated by the following paper:

> K Reynisson, M Schreyer, D Borth (2024). *GraphGuard: Contrastive self-supervised learning for credit-card fraud detection in multi-relational dynamic graphs*. arXiv preprint arXiv:2407.12440. https://arxiv.org/abs/2407.12440

Paper author: Prof. Damian Borth, University of St.Gallen (damian.borth@unisg.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
