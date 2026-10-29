"""Evaluation metrics. Label convention everywhere: 1 = AI-generated, 0 = human-made.

False-positive rate (FPR) = share of HUMAN tracks wrongly called AI. It is the
metric this project cares about most: a false positive accuses a real musician.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = ["balanced_accuracy", "precision", "recall", "fpr", "fnr", "eer", "auc", "accuracy"]
MIN_CLUSTERS = 20  # fewer independent artists than this: no confidence interval is reported


def eer(y_true, p_ai) -> float:
    """Equal error rate: the threshold-free point where FPR equals FNR."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, p_ai)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def summarize(y_true, p_ai, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_ai, dtype=float)
    pred = (p >= threshold).astype(int)
    ai, hu = y == 1, y == 0
    tp, fp = int((pred[ai] == 1).sum()), int((pred[hu] == 1).sum())
    out = {"n": int(len(y)), "n_ai": int(ai.sum()), "n_human": int(hu.sum()), "tp": tp, "fp": fp,
           "fn": int(ai.sum() - tp), "tn": int(hu.sum() - fp)}
    out["recall"] = tp / ai.sum() if ai.any() else np.nan  # = TPR = share of AI tracks caught
    out["fnr"] = 1 - out["recall"] if ai.any() else np.nan
    out["fpr"] = fp / hu.sum() if hu.any() else np.nan
    out["specificity"] = 1 - out["fpr"] if hu.any() else np.nan  # = accuracy on human music
    out["precision"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["accuracy"] = float((pred == y).mean()) if len(y) else np.nan
    out["balanced_accuracy"] = float(np.nanmean([out["recall"], out["specificity"]])) if (ai.any() and hu.any()) else np.nan
    out["auc"] = float(roc_auc_score(y, p)) if (ai.any() and hu.any()) else np.nan
    out["eer"] = eer(y, p)
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def bootstrap_ci(y_true, p_ai, groups, metric: str, n_boot: int = 2000, threshold: float = 0.5, seed: int = 0):
    """95% CI by resampling ARTISTS (not clips), since clips from one artist are not independent.

    Supports the rate metrics: fpr, specificity, recall, fnr, balanced_accuracy.
    Needs at least MIN_CLUSTERS artists (a percentile bootstrap over a handful of clusters
    under-covers). When there are no errors at all (or only errors) the bootstrap collapses to
    zero width, so a Wilson interval over the number of artists is returned instead.
    """
    import pandas as pd

    y, g = np.asarray(y_true).astype(int), np.asarray(groups)
    pred = (np.asarray(p_ai) >= threshold).astype(int)
    c = pd.DataFrame({"g": g, "ai": y, "tp": pred * y, "hu": 1 - y, "fp": pred * (1 - y)}).groupby("g").sum()
    if len(c) < MIN_CLUSTERS:
        return (np.nan, np.nan)
    if metric == "balanced_accuracy":
        degenerate = any(c[n].sum() and c[k].sum() in (0, c[n].sum()) for k, n in (("fp", "hu"), ("tp", "ai")))
        if degenerate:  # average the two components' intervals (each handles its own zero-error case)
            r_lo, r_hi = bootstrap_ci(y_true, p_ai, groups, "recall", n_boot, threshold, seed)
            s_lo, s_hi = bootstrap_ci(y_true, p_ai, groups, "specificity", n_boot, threshold, seed)
            return ((r_lo + s_lo) / 2, (r_hi + s_hi) / 2)
    for k, n in (("fp", "hu"), ("tp", "ai")):
        if metric in {"fp": ("fpr", "specificity"), "tp": ("recall", "fnr")}[k]:
            events, total, groups_n = c[k].sum(), c[n].sum(), int((c[n] > 0).sum())
            if total and events in (0, total):
                lo, hi = wilson_ci(0 if events == 0 else groups_n, groups_n)
                lo, hi = max(0.0, lo), min(1.0, hi)
                rate = (lo, hi)
                return rate if metric in ("fpr", "recall") else (1 - hi, 1 - lo)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(c), size=(n_boot, len(c)))
    s = {k: c[k].to_numpy()[pick].sum(axis=1).astype(float) for k in ("ai", "tp", "hu", "fp")}
    with np.errstate(invalid="ignore", divide="ignore"):
        rec, fpr = s["tp"] / s["ai"], s["fp"] / s["hu"]
    vals = {"recall": rec, "fnr": 1 - rec, "fpr": fpr, "specificity": 1 - fpr, "balanced_accuracy": (rec + 1 - fpr) / 2}[metric]
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))


def prevalence_correction(flag_rate: float, tpr: float, fpr: float) -> float:
    """Rogan-Gladen estimator: true share of AI tracks given an imperfect detector."""
    if tpr - fpr <= 1e-6:
        return float("nan")
    return float(np.clip((flag_rate - fpr) / (tpr - fpr), 0.0, 1.0))
