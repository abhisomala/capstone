"""Offline tests for Sentinel-2 SCL masking + reflectance feature extraction."""

import numpy as np

from src.data.collect_s2 import features_from_patch

BAND_ROLES = {f"sur_refl_b0{i}": r for i, r in
              zip(range(1, 8), ["red", "nir", "blue", "green", "nir08", "swir16", "swir22"])}
SCL_KEEP = [4, 5, 6, 7]


def _patch(scl, band_dn):
    p = {"__scl__": np.array(scl)}
    for slot in BAND_ROLES:
        p[slot] = (np.array(band_dn), 0.0001, -0.1)
    return p


def test_scl_masks_clouds_and_scales_reflectance():
    # 4 pixels: water(6) keep, cloud-high(9) drop, cloud-shadow(3) drop, veg(4) keep.
    scl = [6, 9, 3, 4]
    dn = [2000, 8000, 500, 3000]        # reflectance = dn*1e-4 - 0.1
    feats = features_from_patch(_patch(scl, dn), BAND_ROLES, SCL_KEEP, min_clean=1)
    assert feats["n_clean_pixels"] == 2                       # only the water + veg pixels
    # mean over kept pixels: ((2000*1e-4-0.1)+(3000*1e-4-0.1))/2 = (0.10+0.20)/2 = 0.15
    assert np.isclose(feats["sur_refl_b01"], 0.15)


def test_nodata_pixels_excluded():
    scl = [6, 6, 6]
    dn = [0, 2000, 3000]                # dn==0 is nodata -> excluded even though SCL=water
    feats = features_from_patch(_patch(scl, dn), BAND_ROLES, SCL_KEEP, min_clean=1)
    assert feats["n_clean_pixels"] == 2


def test_too_few_clean_pixels_returns_none():
    scl = [9, 9, 8, 3]                   # all cloud/shadow
    dn = [2000, 2000, 2000, 2000]
    assert features_from_patch(_patch(scl, dn), BAND_ROLES, SCL_KEEP, min_clean=1) is None


def test_rededge_indices():
    from src.data.rededge import rededge_indices
    # NDCI = (RE1 - red)/(RE1 + red); 2BDA = RE1/red
    f = rededge_indices(b04=0.05, b05=0.08, b06=0.06)
    import numpy as np
    assert np.isclose(f["ndci"], 0.03 / 0.13)
    assert np.isclose(f["two_bda"], 0.08 / 0.05)
    assert "three_bda" in f and "mci" in f
