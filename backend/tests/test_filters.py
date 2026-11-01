"""Tests for preprocessing filters."""

import numpy as np
import pytest

from app.services.filters import gaussian, median, bilateral, clahe, histogram_eq, zscore


@pytest.fixture
def sample_gray():
    return np.random.randint(50, 200, (64, 64), dtype=np.uint8)


def test_gaussian_shape(sample_gray):
    result = gaussian(sample_gray, sigma=1.0)
    assert result.shape == sample_gray.shape


def test_gaussian_smooths(sample_gray):
    result = gaussian(sample_gray, sigma=3.0)
    assert result.std() <= sample_gray.std()


def test_median_shape(sample_gray):
    result = median(sample_gray, ksize=5)
    assert result.shape == sample_gray.shape


def test_bilateral_shape(sample_gray):
    result = bilateral(sample_gray)
    assert result.shape == sample_gray.shape


def test_clahe_range(sample_gray):
    result = clahe(sample_gray)
    assert result.min() >= 0
    assert result.max() <= 255


def test_histogram_eq_range(sample_gray):
    result = histogram_eq(sample_gray)
    assert result.dtype == np.uint8


def test_zscore_range(sample_gray):
    result = zscore(sample_gray)
    assert result.min() >= 0
    assert result.max() <= 255


def test_zscore_constant():
    const = np.full((32, 32), 100, dtype=np.uint8)
    result = zscore(const)
    np.testing.assert_array_equal(result, const)
