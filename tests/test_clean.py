"""Offline tests for cloud masking, calibration, and feature extraction."""

import numpy as np

from src.data.clean import (
    cloud_mask_from_state,
    clean_pixel_mask,
    spectral_features,
)


def test_cloud_mask_bit_decoding():
    # 0 = clear, 3 = assumed clear -> keep; 1 = cloudy, 2 = mixed -> drop.
    clear = cloud_mask_from_state([0, 3, 1, 2])
    assert clear.tolist() == [True, True, False, False]


def test_cloud_mask_flags_shadow_internal_adjacent():
    shadow = 1 << 2
    internal_cloud = 1 << 10
    adjacent = 1 << 13
    clear = cloud_mask_from_state([shadow, internal_cloud, adjacent, 0])
    assert clear.tolist() == [False, False, False, True]


def test_clean_pixel_mask_drops_fill_and_scales():
    bands = {
        "sur_refl_b01": [500, -28672, 800],   # middle pixel is fill/no-data
        "sur_refl_b02": [1000, 1000, 20000],  # third pixel out of valid range
    }
    state = [0, 0, 0]                          # all "clear" per QA
    mask, scaled = clean_pixel_mask(bands, state, scale=0.0001)
    assert mask.tolist() == [True, False, False]
    assert np.isclose(scaled["sur_refl_b01"][0], 0.05)   # 500 * 0.0001


def test_spectral_features_ndvi():
    mean = {f"sur_refl_b0{i}": v for i, v in
            zip(range(1, 8), [0.05, 0.25, 0.03, 0.08, 0.1, 0.1, 0.1])}
    feats = spectral_features(mean)
    # NDVI = (b02 - b01)/(b02 + b01) = (0.25-0.05)/(0.25+0.05) = 0.6667
    assert np.isclose(feats["ndvi"], 0.2 / 0.3)
    assert "green_red_ratio" in feats and "green_blue_ndi" in feats
