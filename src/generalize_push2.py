"""Generalization push, round 2 (generalization_push.md follow-up).

Prior session concluded the ~0.58-0.63 within-lake cross-state ceiling is the MODIS signal
itself. This module tests two specific approaches the prior session did NOT try:

  1. suppress_shortcut  — actively remove the between-station "which lake blooms" shortcut by
     dropping features that are near-constant per station (they encode lake identity, not
     date-to-date bloom dynamics). Ranked by within-station variance fraction.
  2. label-noise handling — wider near-threshold exclusion bands than the prior [4,16] µg/L
     pass, and soft per-sample down-weighting near 8 µg/L, in TRAINING.

Every check reports within-lake (bloom-prone) AUC on the fresh holdouts FRESH1 / FRESH2 and
the diagnosis holdout MI_MO, plus a pooled estimate with bootstrap CI — not LOSO alone, since
the prior session showed LOSO can look good then fail on fresh data (resolvable-scoping:
0.699 LOSO -> 0.275 fresh IL).

Run: python -m src.generalize_push2
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import feature_columns
from src.generalize import station_state_map, blend_model, _standardize


def _add_value_ugL(config, df):
    """Attach measured microcystin value_ugL (max per station-date) for label-noise work."""
    lab = pd.read_csv(resolve_path(config["paths"]["labels_csv"]), low_memory=False)
    lab = lab[lab["characteristic"] == "Microcystin"].copy()
    lab["date"] = lab["date"].astype(str)
    val = lab.groupby(["station_id", "date"])["value_ugL"].max().rename("val")
    df = df.copy(); df["date"] = df["date"].astype(str)
    return df.merge(val, on=["station_id", "date"], how="left")


def within_station_variance_fraction(df, feat, min_dates=3):
    """For each feature, fraction of variance that is WITHIN-station (date-to-date) vs total.

    Low fraction (~0) => constant per lake => identity/between-station shortcut feature.
    High fraction => varies date-to-date at a lake => candidate genuine bloom-dynamics signal.
    Computed on stations with >= min_dates observations.
    """
    counts = df.groupby("station_id")["date"].transform("count")
    multi = df[counts >= min_dates]
    out = {}
    for c in feat:
        gvar = multi.groupby("station_id")[c].var(ddof=0)          # within-station variance
        within = float(np.nanmean(gvar))
        between = float(np.nanvar(multi.groupby("station_id")[c].mean(), ddof=0))
        out[c] = within / (within + between) if (within + between) > 0 else 0.0
    return pd.Series(out).sort_values()


def _auc_ci(y, p, seed=1, n=4000):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    bs = [roc_auc_score(y[i], p[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(n))
          if len(np.unique(y[i])) == 2]
    return float(roc_auc_score(y, p)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fresh_within_lake_eval(config, logger, feat, label, train_df=None, sample_weight=None,
                           predict=None):
    """Train on features.csv (12 states) with optional feature set / weights, then report
    within-lake (bloom-prone) AUC on MI_MO, FRESH1, FRESH2, and pooled. Fresh holdouts only.
    """
    train = train_df if train_df is not None else pd.read_csv(resolve_path(config["paths"]["features_csv"]))
    predict = predict or blend_model()
    Xtr = train[feat].to_numpy(float); ytr = train["label"].to_numpy(int)
    holds = {
        "MI_MO": resolve_path(config["paths"]["holdout_state_features_csv"]),
        "FRESH1": resolve_path(config["paths"]["data_dir"]) / "features" / "features_gen_FRESH1.csv",
        "FRESH2": resolve_path(config["paths"]["data_dir"]) / "features" / "features_gen_FRESH2.csv",
    }
    pooled_y, pooled_p = [], []
    per = {}
    for name, path in holds.items():
        h = pd.read_csv(path)
        leak = set(train["station_id"]) & set(h["station_id"])
        if leak:
            raise RuntimeError(f"{name} leaks {len(leak)} stations into training")
        bp = h[h.groupby("station_id")["label"].transform("max") == 1]
        Xtr_s, Xbp = _standardize(Xtr, bp[feat].to_numpy(float))
        p = predict(Xtr_s, ytr, Xbp) if sample_weight is None else predict(Xtr_s, ytr, Xbp, sample_weight)
        y = bp["label"].to_numpy(int)
        a, lo, hi = _auc_ci(y, p)
        per[name] = (a, lo, hi, len(bp), int(y.sum()))
        pooled_y.append(y); pooled_p.append(p)
        logger.info("  [%s] %-7s within-lake AUC=%.3f [%.3f,%.3f] n=%d pos=%d",
                    label, name, a, lo, hi, len(bp), int(y.sum()))
    py = np.concatenate(pooled_y); pp = np.concatenate(pooled_p)
    pa, plo, phi = _auc_ci(py, pp)
    logger.info("  [%s] POOLED within-lake AUC=%.3f [%.3f,%.3f] n=%d pos=%d",
                label, pa, plo, phi, len(py), int(py.sum()))
    return {"per": per, "pooled": (pa, plo, phi, len(py), int(py.sum()))}


def loso_within_lake(config, logger, feat, label, train_mask=None, sample_weight_fn=None,
                     min_test=40):
    """Within-lake (bloom-prone) LOSO across the 12 training states. Dev signal only."""
    df = pd.read_csv(resolve_path(config["paths"]["features_csv"]))
    df["state"] = df["station_id"].map(station_state_map(config))
    y = df["label"].to_numpy(int); state = df["state"].values
    X = df[feat].to_numpy(float)
    bloomprone = (df.groupby("station_id")["label"].transform("max") == 1).values
    predict = blend_model()
    aucs = []
    for s in [x for x in pd.Series(state).value_counts().index
              if (state == x).sum() >= min_test and len(np.unique(y[state == x])) == 2]:
        te = state == s; trn = (~te)
        if train_mask is not None:
            trn = trn & train_mask
        if len(np.unique(y[trn])) < 2:
            continue
        Xtr, Xte = _standardize(X[trn], X[te])
        w = sample_weight_fn(df[trn]) if sample_weight_fn else None
        p = predict(Xtr, y[trn], Xte) if w is None else predict(Xtr, y[trn], Xte, w)
        sub = bloomprone[te]
        if sub.sum() < 15 or len(np.unique(y[te][sub])) < 2:
            continue
        aucs.append(roc_auc_score(y[te][sub], p[sub]))
    m = float(np.mean(aucs)) if aucs else float("nan")
    logger.info("  [%s] LOSO within-lake mean=%.4f over %d states", label, m, len(aucs))
    return m


def main():
    config = load_config()
    logger = get_run_logger("generalize_push2", resolve_path(config["paths"]["logs_dir"]))
    df = pd.read_csv(resolve_path(config["paths"]["features_csv"]))
    feat = feature_columns(df)
    frac = within_station_variance_fraction(df, feat)
    logger.info("=== within-station variance fraction (low = lake-identity shortcut feature) ===")
    for c, v in frac.items():
        logger.info("  %-22s within_frac=%.3f", c, v)


if __name__ == "__main__":
    main()
