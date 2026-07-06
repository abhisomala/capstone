"""Generalization push (generalization_push.md): make the model transfer across STATES.

The 0.761 station-grouped CV was inflated because train/test folds shared the same 12 states.
Leave-one-state-out (LOSO) within those 12 gives ~0.61 mean and as low as 0.46 (OR), so the
model never generalized across states — it fit a regional confound. This module provides:

  * loso_within_12(...)   — the DEVELOPMENT signal: train on 11 states, test on the held-out
                            one, for each of the 12. Cross-state, uses only existing data.
  * collect_holdout_state(name, states) — fetch WQP microcystin + MODIS imagery + features for
                            states never used here, into data/features/features_gen_<name>.csv,
                            for FRESH-HOLDOUT verification.
  * eval_on_holdout(...)  — train on all 12, score a fresh holdout, with bootstrap CI.

Feature transforms and models are passed in so hypotheses (drop region-proxy features, robust
normalisation, simpler model, ...) are tested by the same harness and logged uniformly.

State provenance is tracked so no holdout that informed a decision is reused as "untouched".
Run e.g.:  python -m src.generalize loso
           python -m src.generalize collect IL US:17
"""

import argparse
import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import feature_columns
from src.data.collect import _collect_station
from src.data.clean import build_features

# The 12 states whose stations are in features.csv (the training set). Michigan and Missouri
# were used to DIAGNOSE the 0.51 failure and are therefore contaminated for verification.
TRAINING_STATES = ("OR", "CA", "WA", "KS", "FL", "NY", "OH", "VT", "NE", "IA", "MN", "WI")
DIAGNOSIS_STATES = ("MI", "MO")


def station_state_map(config):
    """station_id -> state, from the per-state raw WQP files."""
    m = {}
    raw_dir = resolve_path(config["paths"]["raw_ground_truth_dir"])
    for f in glob.glob(str(raw_dir / "microcystin_*.csv")):
        state = os.path.basename(f).split("_")[1].split(".")[0]
        d = pd.read_csv(f, low_memory=False, usecols=["MonitoringLocationIdentifier"])
        for s in d["MonitoringLocationIdentifier"].dropna().unique():
            m.setdefault(s, state)
    return m


def blend_model():
    """Factory for the current LR+GBT blend (0.5/0.5), as functions (fit-on-call)."""
    def predict(Xtr, ytr, Xte):
        pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3).fit(Xtr, ytr)
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                            l2_regularization=1.0)
        gb.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, pos_w, 1.0))
        return 0.5 * lr.predict_proba(Xte)[:, 1] + 0.5 * gb.predict_proba(Xte)[:, 1]
    return predict


def _standardize(Xtr, Xte):
    m, s = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
    s[s == 0] = 1.0
    return (Xtr - m) / s, (Xte - m) / s


def loso_within_12(config, logger, feat_cols=None, predict=None, label="baseline",
                   min_test=40, standardize=_standardize):
    """Leave-one-state-out CV across the 12 training states — the cross-state dev signal."""
    df = pd.read_csv(resolve_path(config["paths"]["features_csv"]))
    feat_cols = feat_cols or feature_columns(df)  # before adding 'state' so it isn't a feature
    feat_cols = [c for c in feat_cols if c != "state"]
    df["state"] = df["station_id"].map(station_state_map(config))
    predict = predict or blend_model()
    X = df[feat_cols].to_numpy(float); y = df["label"].to_numpy(int); state = df["state"].values

    logger.info("=== LOSO-within-12 [%s] : %d features ===", label, len(feat_cols))
    rows = []
    eligible = [st for st in pd.Series(state).value_counts().index
                if (state == st).sum() >= min_test and len(np.unique(y[state == st])) == 2]
    for s in eligible:
        te = state == s; trn = ~te
        Xtr, Xte = standardize(X[trn], X[te])
        p = predict(Xtr, y[trn], Xte)
        auc = roc_auc_score(y[te], p)
        rows.append((s, int(te.sum()), float(y[te].mean()), auc))
        logger.info("  hold %-3s n=%4d bloom=%.2f AUC=%.4f", s, int(te.sum()), y[te].mean(), auc)
    aucs = np.array([a for *_, a in rows])
    logger.info("  [%s] LOSO mean=%.4f  std=%.4f  min=%.4f (%s)", label,
                aucs.mean(), aucs.std(), aucs.min(), rows[int(np.argmin(aucs))][0])
    return rows


def _select_state_measurements(config, states, seed, n_neg_stations):
    """Bloom-bearing stations (all dates) + seed-fixed negative-station sample, valid coords."""
    from src.data.ground_truth import fetch_state_results, build_label_frame
    gt = config["ground_truth"]
    frames = []
    for name, code in states.items():
        raw = fetch_state_results(gt["result_endpoint"], code, gt["primary_characteristic"],
                                  gt["date_start"], gt["date_end"])
        if len(raw):
            frames.append(raw)
    lab = build_label_frame(pd.concat(frames, ignore_index=True),
                            float(gt["microcystin_bloom_threshold_ugL"]),
                            gt["microcystin_threshold_reference"], gt["source_name"],
                            gt["result_endpoint"])
    lab = lab.dropna(subset=["latitude", "longitude"])
    lab = lab[(lab["latitude"].abs() > 0.01) & (lab["longitude"].abs() > 0.01)]
    ever = lab.groupby("station_id")["label"].max()
    pos = sorted(ever[ever == 1].index)
    neg_pool = sorted(ever[ever == 0].index)
    neg = list(pd.Series(neg_pool).sample(n=min(n_neg_stations, len(neg_pool)), random_state=seed))
    sel = lab[lab["station_id"].isin(set(pos) | set(neg))].copy()
    cap = config["imagery"]["max_measurements_per_station"]

    def _cap(g):
        if len(g) <= cap:
            return g
        p = g[g["label"] == 1]; ng = g[g["label"] == 0]
        return pd.concat([p, ng.sample(n=max(0, cap - len(p)), random_state=seed) if len(ng) else ng])
    sel = sel.groupby("station_id", group_keys=False)[sel.columns].apply(_cap)
    return sel[["station_id", "date", "latitude", "longitude", "label"]].reset_index(drop=True), len(pos)


def collect_holdout_state(config, logger, name, states, n_neg_stations=60):
    """Collect MODIS features for fresh holdout states -> features_gen_<name>.csv."""
    seed = config["holdout"]["random_seed"]
    data_dir = resolve_path(config["paths"]["data_dir"])
    patches_path = data_dir / "imagery" / f"gen_{name}_patches.jsonl"
    feats_path = data_dir / "features" / f"features_gen_{name}.csv"
    patches_path.parent.mkdir(parents=True, exist_ok=True)

    sel, n_pos_stations = _select_state_measurements(config, states, seed, n_neg_stations)
    logger.info("[collect %s] states=%s: %d measurements, %d stations (%d bloom-bearing), %d pos station-dates",
                name, list(states), len(sel), sel["station_id"].nunique(), n_pos_stations,
                int(sel.groupby(["station_id", "date"])["label"].max().sum()))

    done = set()
    if patches_path.exists():
        for line in open(patches_path):
            try:
                r = json.loads(line); done.add((r["station_id"], r["date"]))
            except json.JSONDecodeError:
                continue
    remaining = sel[~sel.apply(lambda r: (r["station_id"], str(r["date"])) in done, axis=1)]
    stations = list(remaining.groupby("station_id"))
    cfg = config["imagery"]; dates_cache, cl, wl = {}, threading.Lock(), threading.Lock()
    logger.info("[collect %s] fetching %d stations, %d workers...", name, len(stations), cfg["max_workers"])
    written = 0
    with open(patches_path, "a") as out:
        with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
            futs = {ex.submit(_collect_station, sid, rows, cfg, dates_cache, cl, logger): sid
                    for sid, rows in stations}
            for fut in as_completed(futs):
                try:
                    recs = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning("  %s failed: %s", futs[fut], e); continue
                with wl:
                    for rec in recs:
                        out.write(json.dumps(rec) + "\n")
                    out.flush(); written += len(recs)
    logger.info("[collect %s] wrote %d records; building features...", name, written)
    df = build_features(config, logger, patches_path=patches_path, out_path=feats_path)
    logger.info("[collect %s] features: %d rows, %d stations, %d positive -> %s",
                name, len(df), df["station_id"].nunique(), int(df["label"].sum()), feats_path)
    return feats_path


def classify_holdout(config, logger, name):
    """Add a cyan_class column to features_gen_<name>.csv via the NHD waterbody API.

    Station coordinates come from the collected patches file. Lets the fresh holdout be
    evaluated by resolvability class, the same bucketing used on the training set.
    """
    from src.data.waterbody import query_waterbody_area, classify
    data_dir = resolve_path(config["paths"]["data_dir"])
    patches = data_dir / "imagery" / f"gen_{name}_patches.jsonl"
    coords = {}
    for line in open(patches):
        try:
            r = json.loads(line)
            coords[r["station_id"]] = (r["latitude"], r["longitude"])
        except (json.JSONDecodeError, KeyError):
            continue
    wcfg = config["waterbody"]
    ep = wcfg["nhd_waterbody_endpoint"]

    def _cls(sid):
        lat, lon = coords[sid]
        try:
            area, _, _ = query_waterbody_area(ep, lat, lon)
        except Exception:  # noqa: BLE001
            area = None
        return sid, classify(area, wcfg["single_pixel_km2"], wcfg["reliable_resolvable_km2"]), area

    feats_path = data_dir / "features" / f"features_gen_{name}.csv"
    df = pd.read_csv(feats_path)
    sids = [s for s in df["station_id"].unique() if s in coords]
    cls_map, area_map = {}, {}
    with ThreadPoolExecutor(max_workers=wcfg["max_workers"]) as ex:
        for fut in as_completed([ex.submit(_cls, s) for s in sids]):
            sid, cls, area = fut.result()
            cls_map[sid] = cls; area_map[sid] = area
    df["cyan_class"] = df["station_id"].map(cls_map)
    df["area_sqkm"] = df["station_id"].map(area_map)
    df.to_csv(feats_path, index=False)
    vc = df["cyan_class"].value_counts().to_dict()
    logger.info("[classify %s] %s | bloom rate by class: %s", name, vc,
                {c: round(df[df.cyan_class == c]["label"].mean(), 2) for c in vc})
    return df


def eval_on_holdout(config, logger, holdout_name, feat_cols=None, predict=None, label="baseline",
                    standardize=_standardize, n_boot=2000):
    """Train on all 12 (features.csv), score a fresh holdout features_gen_<name>.csv, with CI."""
    data_dir = resolve_path(config["paths"]["data_dir"])
    train = pd.read_csv(resolve_path(config["paths"]["features_csv"]))
    hold = pd.read_csv(data_dir / "features" / f"features_gen_{holdout_name}.csv")
    leak = set(train["station_id"]) & set(hold["station_id"])
    if leak:
        raise RuntimeError(f"Holdout {holdout_name} leaks {len(leak)} stations into train")
    feat_cols = feat_cols or feature_columns(train)
    predict = predict or blend_model()
    Xtr, Xho = standardize(train[feat_cols].to_numpy(float), hold[feat_cols].to_numpy(float))
    p = predict(Xtr, train["label"].to_numpy(int), Xho)
    y = hold["label"].to_numpy(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    # deterministic bootstrap CI (seeded)
    rng = np.random.RandomState(20260705); boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    logger.info("HOLDOUT[%s | %s]: AUC=%.4f 95%%CI=[%.4f,%.4f] n=%d pos=%d neg=%d stations=%d",
                holdout_name, label, auc, lo, hi, len(y), int(y.sum()), int((y == 0).sum()),
                hold["station_id"].nunique())
    return {"auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(y)),
            "n_pos": int(y.sum()), "label": label, "holdout": holdout_name}


def load_broad_training(config, extra_names):
    """Concatenate features.csv (12 states) with extra collected state files (features_gen_*
    / features_holdout_state) into one broader-geography training frame. Deduped by
    (station_id, date). Used to test whether MORE training geography closes the gap.
    """
    data_dir = resolve_path(config["paths"]["data_dir"])
    frames = [pd.read_csv(resolve_path(config["paths"]["features_csv"]))]
    for nm in extra_names:
        if nm in ("MI_MO", "holdout_state"):
            p = resolve_path(config["paths"]["holdout_state_features_csv"])
        else:
            p = data_dir / "features" / f"features_gen_{nm}.csv"
        frames.append(pd.read_csv(p))
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    common = [c for c in frames[0].columns if c in common]
    df = pd.concat([f[common] for f in frames], ignore_index=True)
    return df.drop_duplicates(subset=["station_id", "date"]).reset_index(drop=True)


def eval_holdout_with_training(config, logger, holdout_name, train_df, label,
                               feat_cols=None, predict=None, standardize=_standardize, n_boot=2000):
    """Train on a provided (broad) frame, score a fresh holdout features_gen_<name>.csv, CI."""
    data_dir = resolve_path(config["paths"]["data_dir"])
    hold = pd.read_csv(data_dir / "features" / f"features_gen_{holdout_name}.csv")
    leak = set(train_df["station_id"]) & set(hold["station_id"])
    if leak:
        raise RuntimeError(f"Holdout {holdout_name} leaks {len(leak)} stations into training")
    feat_cols = feat_cols or [c for c in feature_columns(train_df) if c in hold.columns]
    predict = predict or blend_model()
    Xtr, Xho = standardize(train_df[feat_cols].to_numpy(float), hold[feat_cols].to_numpy(float))
    p = predict(Xtr, train_df["label"].to_numpy(int), Xho)
    y = hold["label"].to_numpy(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    rng = np.random.RandomState(20260705); boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    logger.info("HOLDOUT[%s | %s]: AUC=%.4f 95%%CI=[%.4f,%.4f] n=%d pos=%d neg=%d trainN=%d",
                holdout_name, label, auc, lo, hi, len(y), int(y.sum()), int((y == 0).sum()),
                len(train_df))
    return {"auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi), "holdout": holdout_name,
            "label": label, "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["loso", "collect", "eval"])
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()
    config = load_config()
    logger = get_run_logger("generalize", resolve_path(config["paths"]["logs_dir"]))
    if args.cmd == "loso":
        loso_within_12(config, logger)
    elif args.cmd == "collect":
        # Usage: collect <name> <CODE>              (single state, name doubles as the state)
        #        collect <name> <STATE> <CODE> ...  (one or more explicit STATE CODE pairs)
        name, rest = args.rest[0], args.rest[1:]
        if len(rest) == 1:
            states = {name: rest[0]}
        else:
            states = {rest[i]: rest[i + 1] for i in range(0, len(rest), 2)}
        collect_holdout_state(config, logger, name, states)
    elif args.cmd == "eval":
        eval_on_holdout(config, logger, args.rest[0])


if __name__ == "__main__":
    main()
