# Algae Bloom Detection Model

**Harmful algal bloom (HAB) detection from satellite multispectral imagery, trained on real
measured cyanotoxin data — not image heuristics.**

Labels come from lab-measured microcystin concentrations (EPA/USGS Water Quality Portal);
imagery from NASA MODIS. Every number below is produced by the code and written to `logs/`.
This project is deliberately built around **honest measurement**: it reports what the data can
actually support, and says so plainly when that is less than it first appeared.

---

## TL;DR

The pipeline runs end to end and the in-sample cross-validation number looks like a working
detector (**ROC-AUC 0.76**). It isn't. When tested on **entirely new states the model had never
seen**, performance collapses to chance (**0.51**). The 0.76 was inflated because
cross-validation folds shared the same states — the model learned *which lakes tend to bloom*,
not *how to see a bloom in the imagery*. After an extensive, logged investigation, the real,
transferable bloom signal is weak (**cross-state AUC ≈ 0.58–0.64**), and **no modeling change
moved it**. The binding constraint is sensor resolution: MODIS 500 m is mostly mixed land/water
pixels over the small lakes that make up the target.

| What we measured | Result | Verdict |
|---|---|---|
| In-sample CV ROC-AUC (station-grouped 5-fold) | **0.761** (range 0.717–0.787) |  inflated |
| Untouched-holdout ROC-AUC (new states: MI, MO) | **0.512** — 95% CI [0.445, 0.587] | chance-level |
| Honest cross-state signal (leave-one-state-out) | **0.609** all-lakes / **0.63** within-lake | ↓ the true ceiling |
| Best pooled fresh-holdout within-lake AUC | **0.58–0.64** | modest, above chance |

> **Status:** research pipeline with an honest negative headline. It is **not** a deployable
> detector. The value here is a fully reproducible, leakage-audited investigation of *why*, and
> a precise statement of what would be required to move the number.

---

## The idea

Chlorophyll tells you there's *algae*; it can't tell you the algae is *toxic*. This project
labels imagery with **microcystins** — the dominant cyanobacterial toxin — so the target is the
actual public-health hazard, not a greenness proxy.

The intended differentiation is small waterbodies. NOAA's operational **CyAN** product runs on
300 m Sentinel-3 and can only resolve large lakes; in this dataset **81% of bloom events occur
on waterbodies CyAN cannot reliably resolve**. A model that worked on those small lakes would
cover exactly the gap CyAN leaves — which is why the generalization result below matters so much.

---

## Data sources (all public; no authentication required)

| Source | Used for | Auth |
|---|---|---|
| **EPA/USGS Water Quality Portal** (`waterqualitydata.us`) | Microcystin labels | none |
| **NASA MODIS / ORNL** (`modis.ornl.gov`, MOD09A1) | 500 m multispectral imagery | none |
| **NHDPlus V2.1** via EPA EnviroAtlas | Waterbody area → CyAN-resolvability class | none |
| **EPA CyAN REST API** (`cyan.epa.gov`) | Real CyAN cyanobacteria-index for agreement check | none |
| **Sentinel-2 L2A** via AWS Earth Search | Higher-res / red-edge comparison | none |

### How a measurement becomes a label

A station-date is labelled **bloom (1)** when microcystin ≥ **8 µg/L** (U.S. EPA 2019 recreational
criterion). Censored results are treated as intervals — a non-detect or `"<0.3"` bounds the
value to `[0, 0.3]` µg/L (no-bloom); values straddling the threshold are left unlabelled rather
than guessed. Every row keeps full provenance: source, station, coordinates, date, raw value and
unit, converted µg/L with bounds, censoring flag, threshold, and reference.

---

## Results, honestly

### 1 · The headline that doesn't hold up

In-sample, station-grouped 5-fold CV of a logistic-regression + gradient-boosted-tree blend over
49 spectral features gives **ROC-AUC 0.761** (per-fold 0.717–0.787, std 0.025; multi-seed mean
0.757). Per CyAN-resolvability class: resolvable **0.80**, small/unresolvable **0.78**, marginal
**0.61**. *(log: `logs/cross_validate_*.log`)*

### 2 · The untouched-holdout check — run once, reported as-is

Michigan and Missouri were **never collected during any feature engineering**. Collected through
the identical pipeline and scored a single time, the 0.761 model gets **ROC-AUC 0.512**, bootstrap
95% CI **[0.445, 0.587]** — a **0.249** gap that puts 0.761 outside the interval. On that holdout,
bloom and no-bloom predictions average 0.389 vs 0.379: the model isn't separating classes on new
geography. *(logs: `logs/holdout_state_result.json`, `logs/holdout_state_ci.log`)*

### 3 · Why — diagnosis

Decomposing the pooled 0.759 out-of-fold AUC:
- **Within-station signal = 0.607** — can it tell a bloom date from a non-bloom date at the *same*
  lake? Modestly. This is the part that *transfers*.
- **Between-station ranking = 0.512** (Spearman of station mean-prediction vs station bloom-rate)
  — the model learns *which lakes/regions bloom* from lake-constant features. This inflates the
  pooled number and **does not transfer**; on new geography it becomes false alarms on clean lakes.

Leave-one-state-out CV confirms it: mean **0.609**, ranging 0.71 (CA) down to 0.46 (OR). The
0.761 was never a cross-state estimate.

### 4 · Agreement with real CyAN

Sampled the **real** CyAN cyanobacteria-index at each station via the public EPA CyAN API (the
bulk NASA rasters need Earthdata auth; the API serves the same product openly). On resolvable
lakes — the only place CyAN has signal — model-vs-CyAN agreement is **weak** (Spearman 0.15,
n.s.; 66.7% at a hard threshold, but CyAN calls these hypereutrophic lakes "bloom" ~97% of the
time so the threshold agreement saturates). Notably CyAN agrees with the microcystin ground truth
only ~45% — because **CyAN measures cyanobacteria biomass, not the toxin** the labels use.
*(log: `logs/cyan_agreement_*.log`)*

### 5 · The marginal class is a real limit, not label noise

The marginal-resolvability class sits at **0.61**. Only 4% of its samples fall within ±20% of the
8 µg/L threshold, and excluding them leaves AUC at 0.60 — so it is **not** near-threshold label
noise. No dedicated feature, specialist model, or Sentinel-2 red-edge band recovered it.
*(log: `logs/marginal_diagnosis_*.log`)*

### 6 · What was tried to fix generalization (all logged, all negative)

| Lever | Cross-state within-lake AUC | |
|---|---|---|
| Baseline blend (49 features) | 0.63 | — |
| Drop region-proxy / absolute-reflectance features | 0.55–0.61 | ✗ |
| Simpler / harder-regularized models (LR, GBT) | 0.58–0.61 | ✗ |
| Quantile / robust normalization | 0.59 | ✗ |
| State-balanced training | 0.59 | ✗ worse |
| Scope to resolvable lakes | 0.70 LOSO → **0.28 on fresh IL** | ✗ failed on fresh data |
| Within-station "anomaly" features | 0.56 | ✗ |
| Sentinel-2 red-edge / NDCI | 0.61 vs 0.61 base | ✗ no effect |
| Broad training (+16 states) | +0.02 all-lakes only | ✗ marginal |
| Suppress the between-station shortcut directly | 0.64 → 0.55 | ✗ worse |
| Aggressive near-threshold label cleanup + soft weighting | 0.64 → 0.62–0.65 | ✗ flat |

*(logs: `logs/generalize_*.log`, `logs/generalize_push2_*.log`)*

**Conclusion — the blocker is the data, not the model.** MODIS 500 m carries only a weak,
regionally-consistent bloom signal that transfers at ~0.58–0.64 and false-alarms on unfamiliar
clean lakes. Every modeling lever leaves that ceiling untouched. Genuine improvement would need a
different signal — higher-resolution or bloom-specific bands *proven to transfer across regions* —
which nothing tested here provided.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the pipeline

```bash
# 1. Ground truth: pull real microcystin measurements -> labelled data.
.venv/bin/python -m src.data.ground_truth

# 2. Classify each station's waterbody by CyAN resolvability (the differentiation).
.venv/bin/python -m src.data.waterbody

# 3. Imagery: fetch co-located MODIS composites per station-date (resumable).
.venv/bin/python -m src.data.collect

# 4. Cloud-mask + calibrate + extract spectral features.
.venv/bin/python -m src.data.clean

# 5. HEADLINE in-sample metric: station-grouped k-fold CV (AUC as a range).
.venv/bin/python -m src.cross_validate
```

### Reproduce the honest findings

```bash
# Untouched-state holdout (item that breaks the 0.761 story)
.venv/bin/python -m src.holdout state-labels     # pull MI + MO ground truth
.venv/bin/python -m src.holdout state-collect     # MODIS imagery for them
.venv/bin/python -m src.holdout state-features
.venv/bin/python -m src.holdout state-evaluate    # scored exactly once

# Cross-state generalization diagnosis + fix attempts
.venv/bin/python -m src.generalize loso           # leave-one-state-out
.venv/bin/python -m src.generalize_push2          # shortcut & label-noise tests

# Real CyAN agreement (open EPA API, no auth)
.venv/bin/python -m src.cyan_agreement

# Marginal-class label-noise vs model-failure diagnosis
.venv/bin/python -m src.marginal_diagnosis

# Offline unit tests (no network)
.venv/bin/python -m pytest tests/ -q
```

Every region, date, threshold, imagery, split, and training setting lives in
`config/paths.yaml`. There are **no machine-specific paths in `src/`**; all paths resolve against
the repository root. Timestamped logs for every run land in `logs/`.

---

## Repository layout

```
config/paths.yaml            all paths + pipeline config (single source of truth)
src/
  utils/                     config loading (repo-root path resolution), per-run logging
  data/
    ground_truth.py          WQP pull -> measurement-backed labels (censoring-aware)
    waterbody.py             NHDPlus area -> CyAN-resolvability class (differentiation)
    collect.py               MODIS multispectral collection co-located to stations
    collect_s2.py            Sentinel-2 L2A collection (resolution/red-edge comparison)
    clean.py                 cloud masking, calibration, spectral feature extraction
    splits.py / make_splits  site- & time-based leakage-safe splitting
    feature_splits.py        shared deterministic feature loading
  cross_validate.py          HEADLINE: station-grouped k-fold CV, AUC as a range
  compare_sensors.py         MODIS 500 m vs Sentinel-2 on matched samples
  holdout.py                 provably-untouched holdout from never-collected states
  generalize.py              cross-state (leave-one-state-out) diagnosis + fix attempts
  generalize_push2.py        shortcut-suppression & aggressive label-noise tests
  cyan_agreement.py          agreement vs real CyAN cyanobacteria-index (EPA API)
  marginal_diagnosis.py      marginal-class: label noise vs model failure
  label_noise.py             near-threshold label-noise audit
  model_sweep.py             model/feature sweep utility
  models/model.py            regularised MLP (unstable on this data; not headline)
  train.py / evaluate.py     single-split MLP pipeline (not used for headline numbers)
tests/                       offline unit tests (transforms, masking, splitting, CV)
legacy/                      deprecated heuristic pipeline (negative example, not run)
data/  logs/  models/        generated at runtime (gitignored)
```

---

## Guarantees this project holds itself to

- **No leakage.** Splits and folds are grouped by station so no lake appears in both train and
  test; every holdout is asserted to share zero stations with training.
- **No single-run claims.** Reported numbers are cross-validated or given with a bootstrap CI.
  A holdout is scored exactly once and reported as-is, good or bad.
- **No proxies for real data.** The CyAN agreement uses real CyAN values, not a re-implementation.
- **Everything is logged.** If a number isn't in `logs/`, it isn't reported.

## What would move the number

- **Higher-resolution imagery** (Sentinel-2 10–20 m, or Sentinel-3 OLCI red-edge bands tuned to
  cyanobacteria) instead of MODIS 500 m — the single most likely lever, and the one this data
  cannot overcome on its own.
- **Cross-region training breadth** large enough to reduce clean-lake false alarms (broad
  training helped only marginally at the scale tested).
- **A label better matched to what imagery can see** — reflectance predicts biomass far better
  than it predicts the specific toxin the 8 µg/L threshold keys on.
