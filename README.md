# Algae Bloom Detection

Harmful algal bloom (HAB) detection from satellite multispectral imagery, trained on
**real measured cyanotoxin data**, not image heuristics. See `PROJECT_SPEC.md` for the
full goal and milestone checklist, and `legacy/` for the deprecated heuristic version
that this replaces.

> **Status (honest scope).** The pipeline runs end to end: measurement-backed labels →
> co-located MODIS multispectral imagery → cloud masking + calibration → leakage-safe
> station split → model → evaluation. The headline metric is **5-fold station-grouped
> cross-validation ROC-AUC = 0.69 (range 0.63–0.73)** — signal above noise, **below a
> working-detector bar (~0.75)**. The binding constraint is sensor resolution (MODIS
> 500 m); see [`pathtoselleable.md`](pathtoselleable.md) for what would make this a
> product. Every number here is produced by the code and logged to `logs/`.

## What the labels are

Labels come from the **EPA / USGS Water Quality Portal** (`waterqualitydata.us`), a
public REST API with no authentication. The primary analyte is **microcystins**, the
dominant cyanobacterial toxin — a direct measurement of the "harmful" in HAB, unlike
chlorophyll-a which cannot separate harmful cyanobacteria from benign algae.

A measurement is labelled **bloom (1)** when the microcystin concentration is at or
above **8 µg/L** (U.S. EPA 2019 recreational criterion / swimming advisory; the WHO
2021 provisional recreational guideline of 24 µg/L is documented as an alternative in
`config/paths.yaml`). Censored results are handled as intervals — a non-detect or a
`"<0.3"` value bounds the concentration to `[0, 0.3]` µg/L and is labelled no-bloom;
values that straddle the threshold are left unlabelled rather than guessed.

Each output row carries full provenance: source, endpoint, organization, station id and
name, latitude/longitude, date, raw value and unit, converted µg/L (with low/high
bounds and a censoring flag), the threshold, and the threshold reference.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the pipeline

```bash
# 1. Ground truth: pull real measurements and write labelled data.
#    -> data/ground_truth/labels.csv (+ per-state raw CSVs), logs to logs/.
.venv/bin/python -m src.data.ground_truth

# 2. Classify each station's waterbody by CyAN resolvability (the differentiation).
#    -> data/ground_truth/stations_waterbody.csv
.venv/bin/python -m src.data.waterbody

# 3. Imagery: fetch co-located MODIS multispectral composites per station-date.
#    -> data/imagery/patches.jsonl (resumable; ~45-60 min for the current sample).
.venv/bin/python -m src.data.collect

# 4. Cloud-mask + calibrate + extract spectral features (+ CyAN-class annotation).
#    -> data/features/features.csv
.venv/bin/python -m src.data.clean

# 5. HEADLINE metric: station-grouped k-fold cross-validation, AUC as a range.
.venv/bin/python -m src.cross_validate

# (Optional) single-split MLP pipeline. NOTE: the MLP is unstable on this data and a
# guard warns when it collapses; use cross_validate for reported numbers.
.venv/bin/python -m src.train
.venv/bin/python -m src.evaluate

# Tests (offline; no network needed)
.venv/bin/python -m pytest tests/ -q
```

All region, date, threshold, imagery, split, and training settings live in
`config/paths.yaml` — there are no machine-specific paths in `src/`.

### What a run produces (last run, 2026-07-02)

Exact figures for any run are written to timestamped files under `logs/`; the Water
Quality Portal changes over time, so numbers will drift.

- **Ground truth:** 41,725 microcystin measurements across 2,035 stations, 3.5% bloom
  prevalence (12 U.S. states, 2010–2024).
- **CyAN coverage gap:** 81.2% of bloom events are on waterbodies CyAN's 300 m sensor
  cannot reliably resolve (see [`pathtoselleable.md`](pathtoselleable.md)).
- **Imagery + features:** 3,374 co-located MODIS composites → **3,109 clean station-dates
  across 342 stations** after cloud/temporal masking, 42.4% bloom prevalence.
- **Headline — 5-fold station-grouped cross-validation** (logistic regression):

  | metric | value |
  |---|---|
  | ROC-AUC (mean) | 0.69 |
  | ROC-AUC (per-fold range) | 0.63 – 0.73 (std 0.03) |
  | PR-AUC (mean, prevalence 0.42) | 0.60 |

  Per-CyAN-class out-of-fold ROC-AUC: small/unresolvable **0.74**, resolvable 0.69,
  marginal 0.52.

**How to read this honestly.** CV ROC-AUC 0.69 is real signal above noise (0.50) but
**below a working-detector bar (~0.75)** — this is not yet a product. Reporting it as a
5-fold range (0.63–0.73) rather than one number is deliberate: on a single split this same
data gave anywhere from 0.51 to 0.68 depending only on the model, so single-split numbers
here are not trustworthy. The binding constraint is resolution: MODIS 500 m is coarser than
CyAN's own 300 m and mostly land over the small lakes that are 81% of the target. Sentinel-2
(10–20 m) is the next step and is on the critical path. The MLP is unstable on this data and
is not used for headline numbers.

## Repository layout

```
config/paths.yaml          # all paths + pipeline config (region, thresholds, imagery, split, training)
src/
  utils/config.py          # loads config, resolves paths against the repo root
  utils/logging_setup.py   # per-run file logging
  data/ground_truth.py     # WQP pull -> measurement-backed labels
  data/waterbody.py        # NHDPlus waterbody area -> CyAN-resolvability class (differentiation)
  data/collect.py          # MODIS multispectral collection co-located to stations
  data/clean.py            # cloud masking, calibration, spectral feature extraction
  data/splits.py           # site- and time-based leakage-safe splitting
  data/make_splits.py      # writes train/val/test CSVs, asserts no leakage
  data/feature_splits.py   # shared deterministic feature loading + split
  models/model.py          # regularised MLP over spectral features (unstable here; not headline)
  cross_validate.py        # HEADLINE: station-grouped k-fold CV, AUC as a range
  train.py                 # single-split MLP: early stopping, best-checkpoint, degenerate guard
  evaluate.py              # single-split held-out test metrics (+ per-CyAN-class breakdown)
tests/                     # offline unit tests (transforms, masking, splitting, model, CV)
legacy/                    # deprecated heuristic pipeline (negative example, not run)
data/  logs/  models/      # generated at runtime, gitignored
```

## Improvements that would raise the ceiling

The pipeline is correct and honest; these would improve the weak baseline:

- **Higher-resolution imagery** (Sentinel-2 10–20 m, or Sentinel-3 OLCI with red-edge
  bands better suited to cyanobacteria) instead of MODIS 500 m.
- **CyAN cyanobacteria-index rasters** as the spec's preferred #1 label source (needs
  NASA Earthdata credentials), with the in-situ microcystin labels as an independent
  agreement check.
- **Tighter temporal matching** and multi-date compositing to reduce the ≤ 8-day offset.
- **Broader collection** — the collector is resumable and capped only by config, so the
  pilot can be scaled up to the full ground-truth set.
