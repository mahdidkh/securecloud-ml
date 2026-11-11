"""Helpers for the SecureCloud AI anomaly-detection module (baseline, scoring, threshold, metrics)."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


class MahalanobisDetector(BaseEstimator):
    """Statistical BASELINE: Mahalanobis distance to the mean of the normal training events.
    Mean and covariance are learned on normal training rows only (ridge-regularised, pseudo-inverse)."""

    def __init__(self, ridge=1e-6):
        self.ridge = ridge

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + self.ridge * np.eye(X.shape[1])
        self.precision_ = np.linalg.pinv(cov)
        return self

    def score_samples(self, X):  # sklearn convention: higher = more normal
        d = np.asarray(X, dtype=float) - self.mean_
        return -np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision_, d))


def anomaly_score(model, X):
    """Higher = more abnormal (sign of sklearn's score_samples is flipped)."""
    return -model.score_samples(np.asarray(X, dtype=float))


def best_threshold(scores, y, n_candidates=400):
    """Threshold maximising F1 on VALIDATION scores/labels (never call it with test data)."""
    scores = np.asarray(scores, dtype=float); y = np.asarray(y, dtype=int)
    cands = np.unique(np.quantile(scores, np.linspace(0.5, 0.999, n_candidates)))
    best_f1, best_t = -1.0, float(np.median(scores))
    for t in cands:
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def summarize(y, pred, scores=None):
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
           "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
           "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
           "alerts_per_10k": round(float((tp + fp) / len(y) * 10_000), 1)}
    if scores is not None and len(np.unique(y)) == 2:
        out["pr_auc"] = round(float(average_precision_score(y, scores)), 4)
        out["roc_auc"] = round(float(roc_auc_score(y, scores)), 4)
    return out
