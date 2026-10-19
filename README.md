# Label-Supervised vs Label-Free Fraud Detection on Time-Ordered Credit-Card Data

Credit-card fraud is rare, costly, and changes over time, so a detector is only useful if it generalises to transactions that arrive after training. The GraphGuard paper ("GraphGuard: Contrastive self-supervised learning for credit-card fraud detection in multi-relational dynamic graphs", arXiv 2407.12440) argues that self-supervised, label-free anomaly detection can work on transaction networks. This experiment asks a simpler, practical question with a standard tabular benchmark rather than a graph.

**Question:** Can a class-balanced, CV-tuned LightGBM that uses the few confirmed fraud labels beat a label-free unsupervised anomaly detector on time-ordered credit-card fraud detection?

**Dataset:** the OpenML credit-card fraud dataset (ID 1597), with PCA features V1-V28, Time, Amount and a fraud label. Transactions are sorted by Time and split into an earlier training period and a later test period. This is not the paper's own dataset.

**Baseline:** an Isolation Forest fitted on training features only, with no labels. It stands in for the label-free family of methods and is not a re-implementation of GraphGuard.

**Improved model:** a LightGBM classifier with balanced class weights. Its hyperparameters are chosen by a small cross-validated random search on the training period only, using the training fraud labels. The test period is used once for the final comparison.

**Metric:** PR-AUC (average precision) on the test period, where higher is better. ROC-AUC, recall and F1 at a 1% alert rate are secondary metrics.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json`; charts are saved to `figures/`.

## Reference

This project is motivated by the following paper:

> K Reynisson, M Schreyer, D Borth (2024). *GraphGuard: Contrastive self-supervised learning for credit-card fraud detection in multi-relational dynamic graphs*. arXiv preprint arXiv:2407.12440. https://arxiv.org/abs/2407.12440

Paper author: Prof. Damian Borth, University of St.Gallen (damian.borth@unisg.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
