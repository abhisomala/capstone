"""Model / architecture sweep (item 4 in model_quality.md).

Resolution (item 1) and red-edge features (item 3) barely moved the number, so this asks
whether the *model* is the limit: does a non-linear model (gradient-boosted trees, random
forest) beat the logistic-regression floor, and does a class-weight sweep help? Everything
is station-grouped k-fold CV at two fold seeds, reported as a range — no single splits.

Run:
  python -m src.model_sweep                                  # full MODIS dataset (headline)
  python -m src.model_sweep --features data/features/features_s2_full.csv   # S2 + red-edge
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import feature_columns
from src.compare_sensors import station_folds


def _cv_model(df, feat_cols, folds, make_model, class_weight_ratio=None):
    """Station-grouped CV for one model factory; returns fold ROC-AUCs."""
    st = df["station_id"].to_numpy()
    y = df["label"].to_numpy(dtype=int)
    X = df[feat_cols].to_numpy(dtype=np.float64)
    aucs = []
    for fold in folds:
        te = np.array([s in fold for s in st])
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        mean, std = X[tr].mean(0), X[tr].std(0)
        std[std == 0] = 1.0
        Xtr, Xte = (X[tr] - mean) / std, (X[te] - mean) / std
        model = make_model()
        # Trees get class weights via sample_weight (they take no class_weight in ctor here).
        if class_weight_ratio is not None:
            w = np.where(y[tr] == 1, class_weight_ratio, 1.0)
            model.fit(Xtr, y[tr], sample_weight=w)
        else:
            model.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(Xte)[:, 1]))
    return aucs


def sweep(config, logger, features_path=None):
    df = pd.read_csv(features_path or resolve_path(config["paths"]["features_csv"]))
    feat_cols = feature_columns(df)
    seeds = [42, 7]
    n_splits = config["cross_validation"]["n_splits"]
    logger.info("Model sweep on %s", features_path or "full MODIS features")
    logger.info("%d samples, %d stations, %d features, bloom=%.1f%%",
                len(df), df["station_id"].nunique(), len(feat_cols), 100 * df["label"].mean())

    balanced = df["label"].value_counts()
    pos_ratio = float(balanced.get(0, 1)) / float(balanced.get(1, 1))

    models = {
        "logreg (balanced)": (lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), None),
        "logreg (weight x2)": (lambda: LogisticRegression(max_iter=2000, class_weight={0: 1, 1: 2}), None),
        f"logreg (weight x{pos_ratio:.1f})": (
            lambda: LogisticRegression(max_iter=2000, class_weight={0: 1, 1: pos_ratio}), None),
        "hist-GBT": (lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                                            learning_rate=0.05, l2_regularization=1.0), pos_ratio),
        "random-forest": (lambda: RandomForestClassifier(n_estimators=400, max_depth=6,
                                                        class_weight="balanced", random_state=0), None),
    }
    for name, (make, cwr) in models.items():
        means = []
        for seed in seeds:
            folds = station_folds(df["station_id"], n_splits, seed)
            aucs = _cv_model(df, feat_cols, folds, make, class_weight_ratio=cwr)
            a = np.array(aucs)
            means.append(a.mean())
        best = np.mean(means)
        # Report the last seed's folds for the range/std picture.
        a = np.array(aucs)
        logger.info("  %-24s CV mean=%.3f (across seeds) | last-seed folds std=%.3f range=[%.3f, %.3f]",
                    name, best, a.std(), a.min(), a.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None)
    args = parser.parse_args()
    config = load_config()
    logger = get_run_logger("model_sweep", resolve_path(config["paths"]["logs_dir"]))
    sweep(config, logger, resolve_path(args.features) if args.features else None)


if __name__ == "__main__":
    main()
