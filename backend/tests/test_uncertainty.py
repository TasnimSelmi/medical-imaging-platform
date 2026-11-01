"""Tests for uncertainty quantification."""

import numpy as np
import pytest


def _make_dummy_model():
    """Build a tiny U-Net-like model for testing."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(256, 256, 1))
    x = layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inputs, x)


@pytest.fixture(scope="module")
def dummy_model():
    return _make_dummy_model()


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (64, 64), dtype=np.uint8)


def test_mc_dropout_output_shapes(dummy_model, sample_image):
    from app.services.uncertainty import mc_dropout_predict
    mask, std_map, entropy_map, conf = mc_dropout_predict(dummy_model, sample_image, n_passes=3)
    assert mask.shape == sample_image.shape[:2]
    assert std_map.shape == sample_image.shape[:2]
    assert entropy_map.shape == sample_image.shape[:2]


def test_mc_dropout_std_positivity(dummy_model, sample_image):
    from app.services.uncertainty import mc_dropout_predict
    _, std_map, _, _ = mc_dropout_predict(dummy_model, sample_image, n_passes=5)
    assert np.all(std_map >= 0)


def test_mc_dropout_confidence_range(dummy_model, sample_image):
    from app.services.uncertainty import mc_dropout_predict
    _, _, _, conf = mc_dropout_predict(dummy_model, sample_image, n_passes=3)
    assert -0.1 <= conf <= 1.1


def test_mc_dropout_mask_binary(dummy_model, sample_image):
    from app.services.uncertainty import mc_dropout_predict
    mask, _, _, _ = mc_dropout_predict(dummy_model, sample_image, n_passes=3)
    unique = set(np.unique(mask))
    assert unique.issubset({0, 255})


def test_colorize_output():
    from app.services.uncertainty import colorize
    arr = np.random.rand(32, 32).astype(np.float32)
    colored = colorize(arr)
    assert colored.shape == (32, 32, 3)
    assert colored.dtype == np.uint8
