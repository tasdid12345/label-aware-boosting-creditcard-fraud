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

## Reference

This project is motivated by the following paper:

> K Reynisson, M Schreyer, D Borth (2024). *GraphGuard: Contrastive self-supervised learning for credit-card fraud detection in multi-relational dynamic graphs*. arXiv preprint arXiv:2407.12440. https://arxiv.org/abs/2407.12440

Paper author: Prof. Damian Borth, University of St.Gallen (damian.borth@unisg.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
