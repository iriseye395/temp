import math

import numpy as np
import pytest

from src.thresholding import (
    HybridThresholdingModel,
    MIN_SCORE,
    CONFIDENCE,
    DEFAULT_MIN_PVALUE_THRESHOLD,
)


# ------------------------------------------------------------------ #
# Construction & validation
# ------------------------------------------------------------------ #


def test_default_construction():
    model = HybridThresholdingModel()
    assert model.min_pvalue_threshold == DEFAULT_MIN_PVALUE_THRESHOLD
    assert model.confidence() == CONFIDENCE


def test_invalid_min_pvalue():
    with pytest.raises(ValueError):
        HybridThresholdingModel(min_pvalue_threshold=0.0)
    with pytest.raises(ValueError):
        HybridThresholdingModel(min_pvalue_threshold=1.0)


def test_invalid_max_rank_error():
    with pytest.raises(ValueError):
        HybridThresholdingModel(max_rank_error=0.0)


def test_invalid_max_score():
    with pytest.raises(ValueError):
        HybridThresholdingModel(max_score=0.0)


def test_invalid_downsample():
    with pytest.raises(ValueError):
        HybridThresholdingModel(
            downsample_num_samples=2,
            downsample_max_num_observations=2,
        )


# ------------------------------------------------------------------ #
# Log-normal helpers
# ------------------------------------------------------------------ #


def test_log_normal_cdf_monotonic():
    model = HybridThresholdingModel()
    mu, sigma = 0.0, 1.0
    assert model._log_normal_cdf(0.1, mu, sigma) < model._log_normal_cdf(1.0, mu, sigma)
    assert model._log_normal_cdf(1.0, mu, sigma) < model._log_normal_cdf(10.0, mu, sigma)


def test_log_normal_quantile_inverse():
    model = HybridThresholdingModel()
    mu, sigma = 0.5, 0.8
    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        score = model._log_normal_quantile(p, mu, sigma)
        recovered_p = model._log_normal_cdf(score, mu, sigma)
        assert math.isclose(recovered_p, p, rel_tol=1e-5)


# ------------------------------------------------------------------ #
# Update & grade
# ------------------------------------------------------------------ #


def test_grade_below_min_score():
    model = HybridThresholdingModel()
    assert model.grade(0.0) == 0.0
    assert model.grade(MIN_SCORE) == 0.0
    assert model.grade(MIN_SCORE - 0.01) == 0.0


def test_grade_on_empty_model():
    model = HybridThresholdingModel()
    # With no updates, get_rank falls back to 0 or 1 depending on sketch impl
    g = model.grade(1.0)
    assert 0.0 <= g <= 1.0


def test_train_and_grade():
    model = HybridThresholdingModel(
        min_pvalue_threshold=0.95,
        num_log_normal_quantiles=50,
        max_score=50.0,
    )
    # Simulate "normal" RCF scores ~ LogNormal(0.5, 0.5)
    rng = np.random.RandomState(42)
    scores = rng.lognormal(mean=0.5, sigma=0.5, size=1_000)
    model.train(scores)

    # Low score should have low grade
    low_grade = model.grade(0.5)
    assert low_grade == 0.0  # MIN_SCORE = 0.4, but 0.5 might still be below threshold

    # Very high score should have high grade
    high_grade = model.grade(20.0)
    assert high_grade > 0.5

    # Grade should be monotonic-ish: higher score -> higher or equal grade
    g1 = model.grade(5.0)
    g2 = model.grade(10.0)
    g3 = model.grade(30.0)
    assert g1 <= g2 <= g3


def test_update_increases_observations():
    model = HybridThresholdingModel()
    model.update(1.0)
    model.update(2.0)
    model.update(3.0)
    # After updates, grades should differentiate
    g_low = model.grade(0.5)
    g_mid = model.grade(2.0)
    g_high = model.grade(5.0)
    assert g_low <= g_mid <= g_high


def test_downsample_trigger():
    model = HybridThresholdingModel(
        downsample_num_samples=5,
        downsample_max_num_observations=10,
    )
    for i in range(10):
        model.update(float(i))
    # After 10th update, downsample should have triggered
    assert model._num_observations <= 5


# ------------------------------------------------------------------ #
# Serialization round-trip
# ------------------------------------------------------------------ #


def test_serialization_round_trip():
    model = HybridThresholdingModel(min_pvalue_threshold=0.97)
    rng = np.random.RandomState(7)
    for s in rng.lognormal(0.0, 0.5, 200):
        model.update(float(s))

    state = model.to_dict()
    restored = HybridThresholdingModel.from_dict(state)

    assert restored.min_pvalue_threshold == model.min_pvalue_threshold
    assert restored.max_score == model.max_score
    assert restored._num_observations == model._num_observations

    # Grades should be identical after restore
    for score in [0.5, 1.0, 2.0, 5.0, 10.0]:
        assert math.isclose(
            model.grade(score), restored.grade(score), rel_tol=1e-6
        )


def test_serialization_empty_model():
    model = HybridThresholdingModel()
    state = model.to_dict()
    restored = HybridThresholdingModel.from_dict(state)
    assert restored._num_observations == 0


# ------------------------------------------------------------------ #
# Confidence constant
# ------------------------------------------------------------------ #


def test_confidence_constant():
    model = HybridThresholdingModel()
    assert model.confidence() == CONFIDENCE
