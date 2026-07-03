"""Compare MODIS vs Sentinel-2 fold-by-fold on the SAME station-dates (item 2 of the plan).

Isolates the effect of imagery resolution: it intersects the MODIS and Sentinel-2 feature
sets on (station, date), assigns station-grouped CV folds ONCE (so both sensors see the
same fold structure), and reports ROC-AUC per fold + per-CyAN-class for each sensor, at
more than one fold seed (so a jump is not a single-split fluke). Same features, same
labels, same folds — only the sensor differs.

Run: python -m src.compare_sensors
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import feature_columns


def station_folds(stations, n_splits, seed):
    """Deterministically shuffle unique stations and split them into n_splits folds."""
    st = np.array(sorted(set(stations)))
    np.random.default_rng(seed).shuffle(st)
    return [set(a) for a in np.array_split(st, n_splits)]


def cv_on(df, feat_cols, folds):
    """Station-grouped CV given fixed station folds. Returns (fold_aucs, oof_predictions)."""
    st = df["station_id"].to_numpy()
    y = df["label"].to_numpy(dtype=int)
    X = df[feat_cols].to_numpy(dtype=np.float64)
    oof = np.full(len(df), np.nan)
    aucs = []
    for fold in folds:
        te = np.array([s in fold for s in st])
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        mean, std = X[tr].mean(0), X[tr].std(0)
        std[std == 0] = 1.0
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit((X[tr] - mean) / std, y[tr])
        p = clf.predict_proba((X[te] - mean) / std)[:, 1]
        oof[te] = p
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return aucs, oof


def _report(logger, name, df, feat_cols, seeds, n_splits):
    logger.info("--- %s (n=%d, %d stations, bloom=%.1f%%) ---",
                name, len(df), df["station_id"].nunique(), 100.0 * df["label"].mean())
    all_means = []
    last_oof = None
    for seed in seeds:
        folds = station_folds(df["station_id"], n_splits, seed)
        aucs, oof = cv_on(df, feat_cols, folds)
        last_oof = oof
        a = np.array(aucs)
        logger.info("  seed %d: folds=[%s] mean=%.3f std=%.3f range=[%.3f, %.3f]",
                    seed, ", ".join(f"{x:.3f}" for x in aucs), a.mean(), a.std(),
                    a.min(), a.max())
        all_means.append(a.mean())
    logger.info("  mean across seeds: %.3f", np.mean(all_means))
    # Per-CyAN-class on the last seed's out-of-fold predictions.
    if "cyan_class" in df.columns and last_oof is not None:
        y = df["label"].to_numpy(dtype=int)
        for cls in ("resolvable", "marginal", "unresolvable"):
            m = (df["cyan_class"] == cls).to_numpy() & ~np.isnan(last_oof)
            if m.sum() and len(np.unique(y[m])) == 2:
                logger.info("    %-12s n=%d bloom=%d ROC-AUC=%.3f",
                            cls, int(m.sum()), int(y[m].sum()), roc_auc_score(y[m], last_oof[m]))
    return np.mean(all_means)


def compare(config: dict, logger, path_a=None, name_a="A", path_b=None, name_b="B"):
    seeds = [42, 7]
    n_splits = config["cross_validation"]["n_splits"]
    a = pd.read_csv(path_a or resolve_path(config["paths"]["features_csv"]))
    b = pd.read_csv(path_b or resolve_path(config["paths"]["features_s2_csv"]))
    a["date"] = a["date"].astype(str)
    b["date"] = b["date"].astype(str)

    # Intersect on (station, date) so both feature sets describe the exact same samples.
    key = ["station_id", "date"]
    common = a[key].merge(b[key], on=key)
    a_c = a.merge(common, on=key).sort_values(key).reset_index(drop=True)
    b_c = b.merge(common, on=key).sort_values(key).reset_index(drop=True)

    logger.info("Comparing %s vs %s on matched station-dates: %d", name_a, name_b, len(common))
    logger.info("Fold seeds: %s | %d-fold station-grouped CV, logistic regression",
                seeds, n_splits)
    # Each feature set uses its own columns (they may differ, e.g. +red-edge).
    a_mean = _report(logger, name_a, a_c, feature_columns(a_c), seeds, n_splits)
    b_mean = _report(logger, name_b, b_c, feature_columns(b_c), seeds, n_splits)
    logger.info("=== EFFECT: %s %.3f -> %s %.3f (delta %+.3f) ===",
                name_a, a_mean, name_b, b_mean, b_mean - a_mean)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=None, help="features CSV A (default: MODIS)")
    parser.add_argument("--b", default=None, help="features CSV B (default: Sentinel-2)")
    parser.add_argument("--name-a", default="MODIS 500 m")
    parser.add_argument("--name-b", default="Sentinel-2 10-20 m")
    args = parser.parse_args()
    config = load_config()
    logger = get_run_logger("compare_sensors", resolve_path(config["paths"]["logs_dir"]))
    compare(config, logger,
            path_a=resolve_path(args.a) if args.a else None, name_a=args.name_a,
            path_b=resolve_path(args.b) if args.b else None, name_b=args.name_b)


if __name__ == "__main__":
    main()
