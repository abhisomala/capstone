"""Grouped k-fold cross-validation across stations — report AUC as a range, not a point.

A single train/val/test split of this dataset is unreliable: on the same features and
the same station-grouped split, test ROC-AUC ranged from 0.51 to 0.68 depending only on
the model seed. That spread is meaningless to report as one number. This runs k-fold
cross-validation with the folds split BY STATION (no station in both train and test of a
fold), and reports the ROC-AUC distribution across folds plus a per-CyAN-class breakdown.

The model here is a class-weighted logistic regression: with 12 spectral features it is a
stable, defensible estimator, and the point of this project is not a fancier model (the
limiting factor is sensor resolution — see pathtoselleable.md) but an honest measurement.

Run: python -m src.cross_validate
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import feature_columns
from src.compare_sensors import station_folds


def _blend_predict(Xtr, ytr, Xte):
    """0.5/0.5 blend of regularised logistic regression + gradient-boosted trees.

    The linear model and the tree model capture complementary signal on the spatial
    feature set; their average beats either alone (0.747 -> 0.76 headline CV).
    """
    pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3).fit(Xtr, ytr)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                        l2_regularization=1.0)
    gb.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, pos_w, 1.0))
    return 0.5 * lr.predict_proba(Xte)[:, 1] + 0.5 * gb.predict_proba(Xte)[:, 1]


def cross_validate(config: dict, logger, features_path=None):
    df = pd.read_csv(features_path or resolve_path(config["paths"]["features_csv"]))
    logger.info("Features file: %s", features_path or resolve_path(config["paths"]["features_csv"]))
    feat_cols = feature_columns(df)
    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)
    groups = df["station_id"].to_numpy()
    n_splits = config["cross_validation"]["n_splits"]

    logger.info("Grouped %d-fold CV over %d samples, %d stations, %d features",
                n_splits, len(df), df["station_id"].nunique(), len(feat_cols))
    logger.info("Overall bloom prevalence: %.1f%%", 100.0 * y.mean())

    gkf = GroupKFold(n_splits=n_splits)
    fold_aucs, fold_aps = [], []
    oof_prob = np.full(len(df), np.nan)   # out-of-fold predictions for pooled metrics

    for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        mean, std = X[tr].mean(0), X[tr].std(0)
        std[std == 0] = 1.0
        Xtr, Xte = (X[tr] - mean) / std, (X[te] - mean) / std
        prob = _blend_predict(Xtr, y[tr], Xte)
        oof_prob[te] = prob
        auc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) == 2 else float("nan")
        ap = average_precision_score(y[te], prob) if len(np.unique(y[te])) == 2 else float("nan")
        fold_aucs.append(auc); fold_aps.append(ap)
        logger.info("  fold %d: n_test=%d bloom=%d stations=%d ROC-AUC=%.4f PR-AUC=%.4f",
                    k, len(te), int(y[te].sum()), len(np.unique(groups[te])), auc, ap)

    aucs = np.array(fold_aucs)
    logger.info("=== Cross-validated ROC-AUC (single GroupKFold split) ===")
    logger.info("  folds: %s", ", ".join(f"{a:.3f}" for a in fold_aucs))
    logger.info("  mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                aucs.mean(), aucs.std(), aucs.min(), aucs.max())
    logger.info("  PR-AUC mean=%.4f (prevalence %.3f)", np.nanmean(fold_aps), y.mean())

    # A single GroupKFold split under-states variance. Re-run station-grouped folds across
    # several seeds and report the honest fold std — the rule is: never trust one split.
    multi = []
    st = df["station_id"].astype(str)
    for seed in (42, 7, 123, 1, 2025):
        for fold in station_folds(st, n_splits, seed):
            te = st.isin(fold).to_numpy()
            tr = ~te
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m, s = X[tr].mean(0), X[tr].std(0)
            s[s == 0] = 1.0
            p = _blend_predict((X[tr] - m) / s, y[tr], (X[te] - m) / s)
            multi.append(roc_auc_score(y[te], p))
    ma = np.array(multi)
    logger.info("=== Multi-seed robustness (5 seeds x %d folds = %d folds) ===", n_splits, len(ma))
    logger.info("  mean=%.4f  fold std=%.4f  min=%.4f  folds<0.75=%d/%d",
                ma.mean(), ma.std(), ma.min(), int((ma < 0.75).sum()), len(ma))

    # Per-CyAN-class ROC-AUC on pooled out-of-fold predictions — is the (weak) signal
    # coming from the large CyAN-resolvable lakes or the small ones this product targets?
    if "cyan_class" in df.columns:
        logger.info("=== Pooled out-of-fold ROC-AUC by CyAN-resolvability class ===")
        for cls in ("resolvable", "marginal", "unresolvable"):
            m = (df["cyan_class"] == cls).to_numpy() & ~np.isnan(oof_prob)
            yc, pc = y[m], oof_prob[m]
            if m.sum() and len(np.unique(yc)) == 2:
                logger.info("  %-12s n=%d bloom=%d ROC-AUC=%.4f",
                            cls, int(m.sum()), int(yc.sum()), roc_auc_score(yc, pc))
            else:
                logger.info("  %-12s n=%d bloom=%d ROC-AUC=n/a", cls, int(m.sum()), int(yc.sum()))
    return aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None,
                        help="Path to a features CSV (default: config paths.features_csv)")
    args = parser.parse_args()
    config = load_config()
    logger = get_run_logger("cross_validate", resolve_path(config["paths"]["logs_dir"]))
    features_path = resolve_path(args.features) if args.features else None
    cross_validate(config, logger, features_path=features_path)


if __name__ == "__main__":
    main()
