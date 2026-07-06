"""Item 1 (final_verification.md): the untouched-holdout check.

The 0.761 cross-validated ROC-AUC was produced after several rounds of feature
iteration evaluated against the same station-grouped folds. That is a real risk of an
optimistic number. The only honest test is a set of stations that were NEVER part of
any features.csv used during that iteration, scored exactly once with a model trained
on everything that WAS touched.

WHAT "UNTOUCHED" COSTS US HERE (reported plainly, not buried):
The feature-engineering rounds already consumed 233 of the 240 bloom-bearing stations
in the ground-truth set. Only 7 bloom-bearing stations were never touched, and of those:
  * 2 (Carter Lake, NE — the only ones that also carry non-bloom dates, 169 of them)
    have no valid coordinates in the WQP source (latitude/longitude = 0,0). A location
    cannot be invented, so per the project's no-proxy rule they are excluded.
  * 5 are single-measurement stations: one positive date each, zero negative dates.
So the untouched, geolocatable positive signal is exactly 5 samples. A holdout needs
negatives too, so negatives are drawn as a seed-fixed random sample of untouched
never-bloomer stations. The consequence — stated up front — is that this holdout has
~5 positives and is therefore statistically under-powered: it cannot, on its own,
confirm or refute 0.761 at the 0.026 (fold-std) resolution the item asks for. It is run
anyway, exactly once, because a weak untouched number is still more honest than a strong
touched one, and because the item requires the holdout be run and reported as-is.

Steps (each idempotent except the final evaluation, which is one-shot by design):
  python -m src.holdout collect      # fetch MODIS imagery for untouched stations
  python -m src.holdout features     # build features via the identical clean.py pipeline
  python -m src.holdout evaluate     # train on all touched data, score holdout ONCE

The evaluate step refuses to run twice: it writes logs/holdout_result.json and will not
overwrite it. A holdout that gets iterated against stops being a holdout.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.collect import _collect_station
from src.data.clean import build_features
from src.data.feature_splits import feature_columns
from src.cross_validate import _blend_predict

# The 0.761 figure this holdout is checked against. Source of record:
# logs/cross_validate_20260703_000824.log — grouped 5-fold CV, mean=0.7612, std=0.0248.
# The item's trust threshold: a holdout more than one fold-std (0.026) below 0.761 means
# 0.761 is not trustworthy.
CV_REFERENCE_MEAN = 0.7612
# Precise logged fold std is 0.0248; final_verification.md item 1 states the trust
# threshold as 0.026, so that is the decision threshold used here.
CV_FOLD_STD = 0.026


def select_holdout_stations(config: dict, logger) -> pd.DataFrame:
    """Measurements for stations provably absent from every features.csv iteration.

    = all 5 untouched, geolocatable positive-bearing stations (every date they have)
    + a seed-fixed random sample of untouched never-bloomer stations (for negatives).
    """
    labels = pd.read_csv(resolve_path(config["paths"]["labels_csv"]), low_memory=False)
    mc = labels[labels["characteristic"] == "Microcystin"].copy()

    touched = set(pd.read_csv(resolve_path(config["paths"]["features_csv"]))["station_id"])
    ever_bloom = mc.groupby("station_id")["label"].max()

    coords = mc.groupby("station_id")[["latitude", "longitude"]].first()
    has_coords = coords[(coords["latitude"].abs() > 0.01)
                        & (coords["longitude"].abs() > 0.01)].index
    untouched_valid = (set(mc["station_id"]) - touched) & set(has_coords)

    pos_stations = sorted(s for s in untouched_valid if ever_bloom.get(s, 0) == 1)
    neg_pool = sorted(s for s in untouched_valid if ever_bloom.get(s, 0) == 0)

    seed = config["holdout"]["random_seed"]
    n_neg = min(config["holdout"]["n_negative_stations"], len(neg_pool))
    neg_stations = list(pd.Series(neg_pool).sample(n=n_neg, random_state=seed))

    keep = set(pos_stations) | set(neg_stations)
    sel = mc[mc["station_id"].isin(keep)].copy()
    # Cap negatives per station (mirror the main pipeline) so no single station dominates.
    cap = config["imagery"]["max_measurements_per_station"]

    def _cap(group):
        if len(group) <= cap:
            return group
        pos = group[group["label"] == 1]
        neg = group[group["label"] == 0]
        keep_neg = neg.sample(n=max(0, cap - len(pos)), random_state=seed) if len(neg) else neg
        return pd.concat([pos, keep_neg])

    sel = sel.groupby("station_id", group_keys=False)[sel.columns].apply(_cap)
    sel = sel[["station_id", "date", "latitude", "longitude", "label"]].reset_index(drop=True)

    logger.info("Untouched positive-bearing stations (valid coords): %d -> %s",
                len(pos_stations), pos_stations)
    logger.info("Untouched never-bloomer negative pool: %d (sampling %d, seed %d)",
                len(neg_pool), n_neg, seed)
    logger.info("Holdout selection: %d measurements, %d stations, %d positive station-dates",
                len(sel), sel["station_id"].nunique(),
                int(sel.groupby(["station_id", "date"])["label"].max().sum()))
    return sel


def collect_holdout(config: dict, logger):
    """Fetch MODIS imagery for the untouched stations into the holdout patches file."""
    cfg = config["imagery"]
    sel = select_holdout_stations(config, logger)
    patches_path = resolve_path(config["paths"]["holdout_patches"])
    patches_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if patches_path.exists():
        for line in open(patches_path):
            try:
                rec = json.loads(line)
                done.add((rec["station_id"], rec["date"]))
            except json.JSONDecodeError:
                continue
        logger.info("Resuming holdout collection: %d records already present.", len(done))

    remaining = sel[~sel.apply(lambda r: (r["station_id"], str(r["date"])) in done, axis=1)]
    stations = list(remaining.groupby("station_id"))
    dates_cache, cache_lock, write_lock = {}, threading.Lock(), threading.Lock()
    logger.info("Fetching %d holdout stations with up to %d workers...",
                len(stations), cfg["max_workers"])

    written = 0
    with open(patches_path, "a") as out:
        with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
            futures = {ex.submit(_collect_station, sid, rows, cfg, dates_cache, cache_lock, logger): sid
                       for sid, rows in stations}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    records = fut.result()
                except Exception as e:  # noqa: BLE001 - log and continue
                    logger.warning("  holdout station %s failed: %s", sid, e)
                    continue
                with write_lock:
                    for rec in records:
                        out.write(json.dumps(rec) + "\n")
                    out.flush()
                    written += len(records)
    logger.info("Holdout collection wrote %d new records to %s (total %d).",
                written, patches_path, len(done) + written)


def build_holdout_features(config: dict, logger):
    """Run the identical clean.py feature pipeline on the holdout patches."""
    df = build_features(config, logger,
                        patches_path=config["paths"]["holdout_patches"],
                        out_path=config["paths"]["holdout_features_csv"])
    logger.info("Holdout features: %d rows, %d stations, %d positive",
                len(df), df["station_id"].nunique(), int(df["label"].sum()))
    return df


def evaluate_holdout(config: dict, logger):
    """Train the blend on ALL touched features, score the holdout exactly once."""
    result_path = resolve_path(config["paths"]["holdout_result_json"])
    if result_path.exists():
        raise RuntimeError(
            f"{result_path} already exists. The holdout is run exactly once; refusing to "
            f"re-score it. To test a fixed model, define a NEW untouched holdout.")

    train = pd.read_csv(resolve_path(config["paths"]["features_csv"]))
    hold = pd.read_csv(resolve_path(config["paths"]["holdout_features_csv"]))

    # Guarantee no station leaks from train into the holdout.
    leak = set(train["station_id"]) & set(hold["station_id"])
    if leak:
        raise RuntimeError(f"Holdout leakage: {len(leak)} stations in both sets: {sorted(leak)[:5]}")

    feat_cols = feature_columns(train)
    Xtr = train[feat_cols].to_numpy(dtype=np.float64)
    ytr = train["label"].to_numpy(dtype=int)
    Xho = hold[feat_cols].to_numpy(dtype=np.float64)
    yho = hold["label"].to_numpy(dtype=int)

    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std == 0] = 1.0
    prob = _blend_predict((Xtr - mean) / std, ytr, (Xho - mean) / std)

    from sklearn.metrics import roc_auc_score, average_precision_score
    n_pos, n_neg = int(yho.sum()), int((yho == 0).sum())
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
        logger.info("Holdout has only one class (pos=%d neg=%d) — ROC-AUC undefined.", n_pos, n_neg)
    else:
        auc = float(roc_auc_score(yho, prob))
    ap = float(average_precision_score(yho, prob)) if n_pos and n_neg else float("nan")
    gap = CV_REFERENCE_MEAN - auc if not np.isnan(auc) else float("nan")

    logger.info("=== ITEM 1 UNTOUCHED HOLDOUT (scored once) ===")
    logger.info("  holdout: n=%d  positives=%d  negatives=%d  stations=%d",
                len(hold), n_pos, n_neg, hold["station_id"].nunique())
    logger.info("  holdout ROC-AUC = %.4f   PR-AUC = %.4f (prevalence %.3f)",
                auc, ap, yho.mean())
    logger.info("  CV reference    = %.4f (grouped 5-fold mean, fold std %.4f)",
                CV_REFERENCE_MEAN, CV_FOLD_STD)
    logger.info("  GAP (CV - holdout) = %+.4f  [%s the %.3f fold-std threshold]",
                gap, "EXCEEDS" if (not np.isnan(gap) and gap > CV_FOLD_STD) else "within",
                CV_FOLD_STD)
    if not np.isnan(gap) and gap > CV_FOLD_STD:
        logger.info("  -> The holdout drops more than one fold-std below 0.761.")
    logger.info("  POWER CAVEAT: with %d positives the 95%% CI on this AUC is roughly "
                "+/-0.1-0.2; this single number cannot by itself confirm or refute 0.761.", n_pos)

    result = {
        "holdout_roc_auc": auc, "holdout_pr_auc": ap,
        "n_total": int(len(hold)), "n_positive": n_pos, "n_negative": n_neg,
        "n_stations": int(hold["station_id"].nunique()),
        "cv_reference_mean": CV_REFERENCE_MEAN, "cv_fold_std": CV_FOLD_STD,
        "gap_cv_minus_holdout": gap,
        "exceeds_fold_std_threshold": bool(not np.isnan(gap) and gap > CV_FOLD_STD),
        "cv_reference_log": "logs/cross_validate_20260703_000824.log",
    }
    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", result_path)
    return result


# ---------------------------------------------------------------------------
# Powered untouched holdout from ENTIRELY NEW STATES.
#
# The first holdout above (untouched stations within the 12 already-collected states) is
# degenerate: feature iteration already consumed 233/240 bloom-bearing stations, so only 5
# untouched positives exist and none survive MODIS cloud masking -> 0 positives, AUC
# undefined. That is a real result, but it cannot actually test 0.761. Item 1 explicitly
# permits defining a NEW, different, still-untouched holdout — so we pull microcystin
# ground truth from states this project has never touched (Michigan, Missouri), where real
# untouched bloom-bearing stations with valid coordinates exist. Same pipeline, scored once.
# ---------------------------------------------------------------------------

def build_state_holdout_labels(config, logger):
    """Fetch + label microcystin ground truth for the never-collected holdout states."""
    from src.data.ground_truth import fetch_state_results, build_label_frame
    gt = config["ground_truth"]
    frames = []
    for name, code in config["holdout"]["new_states"].items():
        logger.info("Fetching untouched holdout state %s (%s)...", name, code)
        raw = fetch_state_results(gt["result_endpoint"], code, gt["primary_characteristic"],
                                  gt["date_start"], gt["date_end"], logger=logger)
        if len(raw):
            frames.append(raw)
    lab = build_label_frame(pd.concat(frames, ignore_index=True),
                            float(gt["microcystin_bloom_threshold_ugL"]),
                            gt["microcystin_threshold_reference"], gt["source_name"],
                            gt["result_endpoint"], logger=logger)
    lab = lab.dropna(subset=["latitude", "longitude"])
    lab = lab[(lab["latitude"].abs() > 0.01) & (lab["longitude"].abs() > 0.01)].reset_index(drop=True)
    out = resolve_path(config["paths"]["holdout_state_labels_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(out, index=False)
    logger.info("Holdout-state labels: %d measurements, %d stations, %d positive -> %s",
                len(lab), lab["station_id"].nunique(), int(lab["label"].sum()), out)
    return lab


def select_state_holdout(config, logger):
    """All bloom-bearing new-state stations (every date) + a seed-fixed negative sample."""
    lab = pd.read_csv(resolve_path(config["paths"]["holdout_state_labels_csv"]), low_memory=False)
    ever = lab.groupby("station_id")["label"].max()
    pos_stations = sorted(ever[ever == 1].index)
    neg_pool = sorted(ever[ever == 0].index)
    seed = config["holdout"]["random_seed"]
    n_neg = min(config["holdout"]["n_negative_stations_new"], len(neg_pool))
    neg_stations = list(pd.Series(neg_pool).sample(n=n_neg, random_state=seed))
    sel = lab[lab["station_id"].isin(set(pos_stations) | set(neg_stations))].copy()
    cap = config["imagery"]["max_measurements_per_station"]

    def _cap(group):
        if len(group) <= cap:
            return group
        pos = group[group["label"] == 1]
        neg = group[group["label"] == 0]
        keep_neg = neg.sample(n=max(0, cap - len(pos)), random_state=seed) if len(neg) else neg
        return pd.concat([pos, keep_neg])

    sel = sel.groupby("station_id", group_keys=False)[sel.columns].apply(_cap)
    sel = sel[["station_id", "date", "latitude", "longitude", "label"]].reset_index(drop=True)
    logger.info("State holdout selection: %d bloom-bearing stations, %d negative stations, "
                "%d measurements, %d positive station-dates", len(pos_stations), len(neg_stations),
                len(sel), int(sel.groupby(["station_id", "date"])["label"].max().sum()))
    return sel


def collect_state_holdout(config, logger):
    """MODIS imagery for the new-state holdout stations (parallel, resumable)."""
    cfg = config["imagery"]
    sel = select_state_holdout(config, logger)
    patches_path = resolve_path(config["paths"]["holdout_state_patches"])
    patches_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if patches_path.exists():
        for line in open(patches_path):
            try:
                done.add((json.loads(line)["station_id"], json.loads(line)["date"]))
            except json.JSONDecodeError:
                continue
        logger.info("Resuming state-holdout collection: %d records present.", len(done))
    remaining = sel[~sel.apply(lambda r: (r["station_id"], str(r["date"])) in done, axis=1)]
    stations = list(remaining.groupby("station_id"))
    dates_cache, cache_lock, write_lock = {}, threading.Lock(), threading.Lock()
    logger.info("Fetching %d state-holdout stations with up to %d workers...",
                len(stations), cfg["max_workers"])
    written = 0
    with open(patches_path, "a") as out:
        with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
            futures = {ex.submit(_collect_station, sid, rows, cfg, dates_cache, cache_lock, logger): sid
                       for sid, rows in stations}
            for fut in as_completed(futures):
                try:
                    records = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning("  state-holdout station %s failed: %s", futures[fut], e)
                    continue
                with write_lock:
                    for rec in records:
                        out.write(json.dumps(rec) + "\n")
                    out.flush()
                    written += len(records)
    logger.info("State-holdout collection wrote %d new records (total %d).", written, len(done) + written)


def build_state_holdout_features(config, logger):
    df = build_features(config, logger,
                        patches_path=config["paths"]["holdout_state_patches"],
                        out_path=config["paths"]["holdout_state_features_csv"])
    logger.info("State-holdout features: %d rows, %d stations, %d positive",
                len(df), df["station_id"].nunique(), int(df["label"].sum()))
    return df


def evaluate_state_holdout(config, logger):
    """Train the blend on ALL touched features, score the new-state holdout exactly once."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    result_path = resolve_path(config["paths"]["holdout_state_result_json"])
    if result_path.exists():
        raise RuntimeError(f"{result_path} already exists. This holdout is run exactly once; "
                           f"refusing to re-score it.")
    train = pd.read_csv(resolve_path(config["paths"]["features_csv"]))
    hold = pd.read_csv(resolve_path(config["paths"]["holdout_state_features_csv"]))
    leak = set(train["station_id"]) & set(hold["station_id"])
    if leak:
        raise RuntimeError(f"Holdout leakage: {len(leak)} stations in both sets.")
    feat_cols = feature_columns(train)
    Xtr = train[feat_cols].to_numpy(float); ytr = train["label"].to_numpy(int)
    Xho = hold[feat_cols].to_numpy(float); yho = hold["label"].to_numpy(int)
    mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1.0
    prob = _blend_predict((Xtr - mean) / std, ytr, (Xho - mean) / std)
    n_pos, n_neg = int(yho.sum()), int((yho == 0).sum())
    auc = float(roc_auc_score(yho, prob)) if n_pos and n_neg else float("nan")
    ap = float(average_precision_score(yho, prob)) if n_pos and n_neg else float("nan")
    gap = CV_REFERENCE_MEAN - auc if not np.isnan(auc) else float("nan")
    logger.info("=== ITEM 1 UNTOUCHED HOLDOUT — NEW STATES MI+MO (scored once) ===")
    logger.info("  holdout: n=%d  positives=%d  negatives=%d  stations=%d",
                len(hold), n_pos, n_neg, hold["station_id"].nunique())
    logger.info("  holdout ROC-AUC = %.4f   PR-AUC = %.4f (prevalence %.3f)", auc, ap, yho.mean())
    logger.info("  CV reference    = %.4f (grouped 5-fold mean, fold-std threshold %.3f)",
                CV_REFERENCE_MEAN, CV_FOLD_STD)
    logger.info("  GAP (CV - holdout) = %+.4f  [%s the %.3f fold-std threshold]", gap,
                "EXCEEDS" if (not np.isnan(gap) and gap > CV_FOLD_STD) else "within", CV_FOLD_STD)
    if not np.isnan(gap) and gap > CV_FOLD_STD:
        logger.info("  -> Holdout drops more than one fold-std below 0.761: 0.761 is NOT trustworthy.")
    else:
        logger.info("  -> Holdout is within one fold-std of 0.761: 0.761 survives this untouched check.")
    result = {"holdout_roc_auc": auc, "holdout_pr_auc": ap, "n_total": int(len(hold)),
              "n_positive": n_pos, "n_negative": n_neg, "n_stations": int(hold["station_id"].nunique()),
              "cv_reference_mean": CV_REFERENCE_MEAN, "cv_fold_std": CV_FOLD_STD,
              "gap_cv_minus_holdout": gap,
              "exceeds_fold_std_threshold": bool(not np.isnan(gap) and gap > CV_FOLD_STD),
              "states": list(config["holdout"]["new_states"].keys()),
              "cv_reference_log": "logs/cross_validate_20260703_000824.log"}
    result_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", result_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["collect", "features", "evaluate", "select",
                                         "state-labels", "state-collect", "state-features",
                                         "state-evaluate"])
    args = parser.parse_args()
    config = load_config()
    logger = get_run_logger("holdout", resolve_path(config["paths"]["logs_dir"]))
    if args.step == "select":
        select_holdout_stations(config, logger)
    elif args.step == "collect":
        collect_holdout(config, logger)
    elif args.step == "features":
        build_holdout_features(config, logger)
    elif args.step == "evaluate":
        evaluate_holdout(config, logger)
    elif args.step == "state-labels":
        build_state_holdout_labels(config, logger)
    elif args.step == "state-collect":
        collect_state_holdout(config, logger)
    elif args.step == "state-features":
        build_state_holdout_features(config, logger)
    elif args.step == "state-evaluate":
        evaluate_state_holdout(config, logger)


if __name__ == "__main__":
    main()
