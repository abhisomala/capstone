# Legacy — deprecated heuristic pipeline (retained as a negative example)

Everything in this folder is the **original** version of the project. It is kept
for the record and as a documented example of what *not* to do. **None of it is part
of the runnable pipeline** and it should not be used to train or evaluate any model.

Why it is disqualifying:

- **`labeling.py`** assigns bloom / no-bloom labels with an HSV green-pixel-ratio
  threshold applied *to the images themselves*. That is not measured bloom data — a
  model trained on these labels just re-learns the threshold rule. Worse, the images
  are single-band MODIS reflectance rendered through the `viridis` colormap, so
  "green" pixels do not even correspond to vegetation/algae.
- **`data_collection` / `datacollectionpt2`** pull a *single* MODIS band for one fixed
  latitude/longitude, then colormap it to an RGBA PNG. No multispectral information
  survives, and the 10,000-image `algae_bloom_dataset/` does not correspond to these
  scripts' output.
- **`dataset.py`** does a random per-image 80/20 split (train/test only, no validation),
  which leaks near-duplicate same-site scenes across the split.
- **`train.py`** keeps the last checkpoint after a fixed 20 epochs (no early stopping,
  no best-checkpoint selection), contains a leftover `breakpoint()`, and has a
  hand-pasted results block in its docstring.
- **`evaluate.py`** reports only loss and raw accuracy on an imbalanced label set.
- **`data_analysis.py`** plots numbers hand-copied from console output.
- Hardcoded Windows paths (`C:/Users/somal/...`) throughout.

The real pipeline replaces all of this — see the top-level `README.md`. Files here are
preserved as originally written (authorship comments intact) and are intentionally not
modified to run.

`labels.csv` in this folder and `../algae_bloom_dataset/` are the heuristic-labeled
dataset, kept only as this negative example.
