"""Label-noise audit (item 5): is the ceiling near-threshold ambiguity, not model capacity?

The label is microcystin >= 8 ug/L. A measurement of 7 vs 9 ug/L is nearly identical water
but gets the opposite label, and the toxin assay itself is noisy near the threshold — so
samples close to 8 inject irreducible label noise. This drops the ambiguous band around the
threshold (keep clear positives >= hi, clear negatives <= lo), relabels the clear cases, and
re-runs station-grouped CV. If the number jumps, the ceiling was label noise, not the model.

Reports a CV range across folds and seeds, and the per-CyAN-class breakdown.

Run: python -m src.label_noise                                        # S2 + red-edge features
     python -m src.label_noise --features data/features/features.csv  # full MODIS
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.compare_sensors import station_folds, cv_on
from src.data.feature_splits import feature_columns


def audit(config, logger, features_path=None):
    feats_path = features_path or resolve_path(config["paths"]["features_s2_csv"])
    df = pd.read_csv(feats_path)
    df["date"] = df["date"].astype(str)

    # Join the measured microcystin value (max per station-date) so we can see how far each
    # sample is from the 8 ug/L threshold.
    lab = pd.read_csv(resolve_path(config["paths"]["labels_csv"]), low_memory=False)
    lab["date"] = lab["date"].astype(str)
    val = lab.groupby(["station_id", "date"])["value_ugL"].max().reset_index()
    df = df.merge(val, on=["station_id", "date"], how="left")
    df = df.dropna(subset=["value_ugL"])
    feat_cols = [c for c in feature_columns(df) if c != "value_ugL"]
    seeds = [42, 7, 123]
    n_splits = config["cross_validation"]["n_splits"]

    logger.info("Label-noise audit on %s (%d samples, %d stations, %d features)",
                feats_path, len(df), df["station_id"].nunique(), len(feat_cols))

    def run_cv(sub, label):
        allf = []
        oof_last = None
        for seed in seeds:
            folds = station_folds(sub["station_id"], n_splits, seed)
            aucs, oof = cv_on(sub, feat_cols, folds)
            oof_last = oof
            allf += aucs
        a = np.array(allf)
        logger.info("  %-34s N=%d st=%d bloom=%.1f%% | CV mean=%.3f std=%.3f below0.75=%d/%d",
                    label, len(sub), sub["station_id"].nunique(), 100 * sub["label"].mean(),
                    a.mean(), a.std(), int((a < 0.75).sum()), len(a))
        return a, oof_last

    logger.info("=== Full label (all cases, >=8 ug/L threshold) ===")
    run_cv(df.assign(label=(df["value_ugL"] >= 8).astype(int)), "all cases, all days")
    d3 = df[df["days_off"] <= 3]
    run_cv(d3.assign(label=(d3["value_ugL"] >= 8).astype(int)), "all cases, <=3d")

    logger.info("=== Clear cases (drop the ambiguous band around 8 ug/L) ===")
    for lo, hi in [(2, 20), (4, 16)]:
        clear = df[(df["value_ugL"] <= lo) | (df["value_ugL"] >= hi)].copy()
        clear["label"] = (clear["value_ugL"] >= hi).astype(int)
        run_cv(clear, f"drop {lo}-{hi} ug/L, all days")
        c3 = clear[clear["days_off"] <= 3]
        a, oof = run_cv(c3, f"drop {lo}-{hi} ug/L, <=3d")
        if lo == 2 and "cyan_class" in c3.columns and oof is not None:
            y = c3["label"].to_numpy()
            logger.info("    per-CyAN-class (drop 2-20, <=3d, seed 123 OOF):")
            for cls in ("resolvable", "marginal", "unresolvable"):
                m = (c3["cyan_class"] == cls).to_numpy() & ~np.isnan(oof)
                if m.sum() and len(np.unique(y[m])) == 2:
                    logger.info("      %-12s n=%d AUC=%.3f", cls, int(m.sum()), roc_auc_score(y[m], oof[m]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=None)
    args = parser.parse_args()
    config = load_config()
    logger = get_run_logger("label_noise", resolve_path(config["paths"]["logs_dir"]))
    audit(config, logger,
          resolve_path(args.features) if args.features else
          resolve_path(config["paths"]["data_dir"]) / "features" / "features_s2_full.csv")


if __name__ == "__main__":
    main()
