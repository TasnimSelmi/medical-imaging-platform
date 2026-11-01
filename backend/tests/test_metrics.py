"""Tests for evaluation metrics."""

import numpy as np
import pytest

from app.services.metrics import iou, dice, pearson_correlation, mse, nmi


def _mask(val: int, shape=(64, 64)) -> np.ndarray:
    return np.full(shape, val, dtype=np.uint8)


def test_iou_perfect():
    m = _mask(255)
    assert iou(m, m) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou(_mask(255), _mask(0)) == pytest.approx(0.0)


def test_iou_partial():
    truth = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    truth[:5, :] = 255
    pred[:3, :] = 255
    assert 0.0 < iou(truth, pred) < 1.0


def test_dice_perfect():
    m = _mask(255)
    assert dice(m, m) == pytest.approx(1.0)


def test_dice_empty():
    assert dice(_mask(0), _mask(0)) == pytest.approx(1.0)


def test_pearson_identical():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    assert pearson_correlation(img, img) == pytest.approx(1.0, abs=1e-6)


def test_mse_identical():
    img = _mask(128)
    assert mse(img, img) == pytest.approx(0.0)


def test_mse_different():
    assert mse(_mask(0), _mask(255)) > 0


def test_iou_disjoint_masks():
    truth = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    truth[:5, :] = 255
    pred[5:, :] = 255
    assert iou(truth, pred) == pytest.approx(0.0)


def test_dice_disjoint():
    truth = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    truth[:5, :] = 255
    pred[5:, :] = 255
    assert dice(truth, pred) == pytest.approx(0.0)


def test_nmi_identical():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    val = nmi(img, img)
    assert val >= 1.0


def test_nmi_identical_is_max():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    other = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    val_self = nmi(img, img)
    val_other = nmi(img, other)
    assert val_self >= val_other
