"""Tests for CyAN-resolvability classification of waterbodies."""

import numpy as np

from src.data.waterbody import classify

SINGLE_PIXEL = 0.09   # km2, one CyAN 300 m pixel
RELIABLE = 1.0        # km2


def test_missing_area_is_unresolvable():
    # No NHD polygon at the point (river / tiny pond) -> CyAN cannot see it.
    assert classify(None, SINGLE_PIXEL, RELIABLE) == "unresolvable"
    # Regression: float NaN must be treated like None, not silently "resolvable".
    assert classify(np.nan, SINGLE_PIXEL, RELIABLE) == "unresolvable"
    assert classify(float("nan"), SINGLE_PIXEL, RELIABLE) == "unresolvable"


def test_area_buckets():
    assert classify(0.05, SINGLE_PIXEL, RELIABLE) == "unresolvable"   # < 1 pixel
    assert classify(0.5, SINGLE_PIXEL, RELIABLE) == "marginal"        # few pixels
    assert classify(7.2, SINGLE_PIXEL, RELIABLE) == "resolvable"      # large lake


def test_boundaries():
    assert classify(0.09, SINGLE_PIXEL, RELIABLE) == "marginal"       # exactly 1 pixel
    assert classify(1.0, SINGLE_PIXEL, RELIABLE) == "resolvable"      # exactly reliable
