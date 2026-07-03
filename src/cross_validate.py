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

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import feature_columns


def cross_validate(config: dict, logger):
    df = pd.read_csv(resolve_path(config["paths"]["features_csv"]))
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
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(Xtr, y[tr])
        prob = model.predict_proba(Xte)[:, 1]
        oof_prob[te] = prob
        auc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) == 2 else float("nan")
        ap = average_precision_score(y[te], prob) if len(np.unique(y[te])) == 2 else float("nan")
        fold_aucs.append(auc); fold_aps.append(ap)
        logger.info("  fold %d: n_test=%d bloom=%d stations=%d ROC-AUC=%.4f PR-AUC=%.4f",
                    k, len(te), int(y[te].sum()), len(np.unique(groups[te])), auc, ap)

    aucs = np.array(fold_aucs)
    logger.info("=== Cross-validated ROC-AUC ===")
    logger.info("  folds: %s", ", ".join(f"{a:.3f}" for a in fold_aucs))
    logger.info("  mean=%.4f  std=%.4f  range=[%.4f, %.4f]",
                aucs.mean(), aucs.std(), aucs.min(), aucs.max())
    logger.info("  PR-AUC mean=%.4f (prevalence %.3f)", np.nanmean(fold_aps), y.mean())

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
    config = load_config()
    logger = get_run_logger("cross_validate", resolve_path(config["paths"]["logs_dir"]))
    cross_validate(config, logger)


if __name__ == "__main__":
    main()
