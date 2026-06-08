import math

import numpy as np
import pytest

from src.models import AnomalyDetector, Feature, IntervalTimeConfiguration
from src.rcf_wrapper import RCFModel, AnomalyDescriptor, _flatten_attribution


def make_detector(**kwargs) -> AnomalyDetector:
    defaults = dict(
        name="test-detector",
        time_field="@timestamp",
        indices=["logs-*"],
        features=[
            Feature(name="cpu", aggregation={"avg": {"field": "cpu"}}),
            Feature(name="mem", aggregation={"avg": {"field": "memory"}}),
        ],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
        detector_id="det-123",
    )
    defaults.update(kwargs)
    return AnomalyDetector(**defaults)


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #


def test_rcf_model_construction():
    detector = make_detector()
    model = RCFModel(detector)
    assert model.dimensions == detector.base_dimension * detector.shingle_size
    assert model.shingle_size == detector.shingle_size
    assert model.total_updates == 0
    assert not model.is_output_ready()


def test_rcf_model_construction_multi_entity():
    detector = make_detector(shingle_size=4)
    model = RCFModel(detector, random_seed=42)
    assert model.dimensions == detector.base_dimension * detector.shingle_size
    assert model.shingle_size == 4


# ------------------------------------------------------------------ #
# Training / cold start
# ------------------------------------------------------------------ #


def test_train_populates_scores():
    detector = make_detector()
    model = RCFModel(detector, random_seed=42)

    rng = np.random.RandomState(42)
    data = rng.normal(loc=0.0, scale=1.0, size=(200, detector.base_dimension))
    model.train(data)

    assert model.total_updates >= 200
    assert model.is_output_ready()
    # Thresholding model should have been trained
    assert model._thresholding_model._num_observations > 0


def test_train_invalid_shape():
    detector = make_detector()
    model = RCFModel(detector)
    with pytest.raises(ValueError):
        model.train(np.array([1.0, 2.0, 3.0]))  # 1-D


# ------------------------------------------------------------------ #
# Streaming process
# ------------------------------------------------------------------ #


def test_process_before_output_ready():
    detector = make_detector()
    model = RCFModel(detector, random_seed=42, min_samples=32)

    for i in range(10):
        desc = model.process([1.0, 2.0])
        assert isinstance(desc, AnomalyDescriptor)
        assert not desc.is_output_ready
        assert desc.total_updates == i + 1


def test_process_after_output_ready():
    detector = make_detector()
    model = RCFModel(detector, random_seed=42, min_samples=5)

    # Prime the model
    rng = np.random.RandomState(7)
    for _ in range(64):
        model._forest.update(rng.normal(size=detector.base_dimension).tolist())

    assert model.is_output_ready()

    desc = model.process([10.0, 20.0])
    assert desc.is_output_ready
    assert desc.rcf_score > 0.0
    assert desc.confidence > 0.0
    assert desc.number_of_trees == 50
    assert desc.relative_index == 0


def test_process_grade_increases_for_anomaly():
    detector = make_detector()
    model = RCFModel(detector, random_seed=42, min_samples=10)

    # Train with normal data
    rng = np.random.RandomState(1)
    for _ in range(500):
        model.process(rng.normal(loc=[5.0, 10.0], scale=0.5, size=2).tolist())

    assert model.is_output_ready()

    # Normal point should have low/zero grade
    normal_desc = model.process([5.2, 10.1])
    # Anomalous point should have higher grade
    anom_desc = model.process([50.0, 100.0])

    assert anom_desc.rcf_score > normal_desc.rcf_score
    assert anom_desc.anomaly_grade >= normal_desc.anomaly_grade


def test_process_batch():
    detector = make_detector()
    model = RCFModel(detector, random_seed=42, min_samples=10)

    rng = np.random.RandomState(3)
    data = rng.normal(loc=[0.0, 1.0], scale=0.3, size=(100, 2))
    results = model.process_batch(data)

    assert len(results) == 100
    ready_count = sum(1 for r in results if r.is_output_ready)
    assert ready_count > 50  # Most should be ready after first 10


# ------------------------------------------------------------------ #
# Preview scoring (no mutation)
# ------------------------------------------------------------------ #


def test_preview_results_no_mutation():
    detector = make_detector()
    model = RCFModel(detector, random_seed=42, min_samples=10)

    # Initial training
    rng = np.random.RandomState(5)
    train_data = rng.normal(size=(200, 2))
    model.train(train_data)

    updates_before = model.total_updates
    data = rng.normal(size=(50, 2))
    results = model.get_preview_results(data)

    assert len(results) == 50
    assert model.total_updates == updates_before  # Unchanged


# ------------------------------------------------------------------ #
# Checkpoint round-trip
# ------------------------------------------------------------------ #


def test_checkpoint_round_trip():
    detector = make_detector(shingle_size=2)
    model = RCFModel(detector, random_seed=42, min_samples=32)

    rng = np.random.RandomState(9)
    for _ in range(200):
        model.process(rng.normal(loc=[5.0, 10.0], scale=1.0, size=2).tolist())

    assert model.is_output_ready()
    # Score a fixed point without updating (use internal forest directly)
    score_before = model._forest.score([50.0, 100.0])

    checkpoint = model.to_dict()
    restored = RCFModel.from_dict(checkpoint, detector)

    assert restored.total_updates == model.total_updates
    assert restored.is_output_ready()
    score_after = restored._forest.score([50.0, 100.0])
    assert math.isclose(score_before, score_after, rel_tol=1e-5)


# ------------------------------------------------------------------ #
# Attribution helper
# ------------------------------------------------------------------ #


def test_flatten_attribution():
    attr = {"high": [0.1, 0.2, 0.3], "low": [0.01, 0.02, 0.03]}
    flat = _flatten_attribution(attr, 3)
    assert flat == pytest.approx([0.11, 0.22, 0.33])


def test_flatten_attribution_pad():
    attr = {"high": [0.1], "low": [0.01]}
    flat = _flatten_attribution(attr, 4)
    assert flat[0] == 0.11
    assert flat[1] == 0.0
    assert flat[2] == 0.0
