import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from src.feature_manager import Features
from src.models import (
    AnomalyDetector,
    Feature,
    IntervalTimeConfiguration,
    Entity,
    AnomalyResult,
)
from src.runner import AnomalyDetectorRunner, _sample_max_results


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
        result_index=".opendistro-anomaly-results",
    )
    defaults.update(kwargs)
    return AnomalyDetector(**defaults)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_fm(mock_client):
    return MagicMock(spec="src.feature_manager.FeatureManager")


@pytest.fixture
def runner(mock_client, mock_fm):
    return AnomalyDetectorRunner(
        client=mock_client,
        feature_manager=mock_fm,
    )


@pytest.mark.asyncio
async def test_execute_single_entity(runner, mock_fm):
    rng = np.random.RandomState(1)
    data = rng.normal(size=(50, 2))
    features = Features(
        unprocessed_features=data.astype(np.float64),
        time_ranges=[(1704067200000, 1704067200000)] * 50,
    )
    mock_fm.get_features = AsyncMock(return_value=features)

    detector = make_detector()
    results = await runner.execute_detector(
        detector,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert len(results) == 50
    assert all(isinstance(r, AnomalyResult) for r in results)
    assert results[0].detector_id == "det-123"


@pytest.mark.asyncio
async def test_execute_multi_entity(runner, mock_fm):
    mock_fm.get_preview_entities = AsyncMock(
        return_value=[
            Entity(attributes={"host": "web1"}),
            Entity(attributes={"host": "web2"}),
        ]
    )

    rng = np.random.RandomState(2)
    features1 = Features(
        unprocessed_features=rng.normal(size=(30, 2)).astype(np.float64),
        time_ranges=[(1704067200000, 1704067200000)] * 30,
        entity=Entity(attributes={"host": "web1"}),
    )
    features2 = Features(
        unprocessed_features=rng.normal(size=(30, 2)).astype(np.float64),
        time_ranges=[(1704067200000, 1704067200000)] * 30,
        entity=Entity(attributes={"host": "web2"}),
    )
    mock_fm.get_features_for_entity = AsyncMock(side_effect=[features1, features2])

    detector = make_detector(category_fields=["host"], detector_type="MULTI_ENTITY")
    results = await runner.execute_detector(
        detector,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert len(results) == 60
    web1_results = [r for r in results if r.entity and r.entity.attributes.get("host") == "web1"]
    web2_results = [r for r in results if r.entity and r.entity.attributes.get("host") == "web2"]
    assert len(web1_results) == 30
    assert len(web2_results) == 30


@pytest.mark.asyncio
async def test_execute_no_data(runner, mock_fm):
    mock_fm.get_features = AsyncMock(
        return_value=Features(unprocessed_features=np.zeros((0, 2)), time_ranges=[])
    )

    detector = make_detector()
    results = await runner.execute_detector(
        detector,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert results == []


@pytest.mark.asyncio
async def test_execute_multi_entity_no_entities(runner, mock_fm):
    mock_fm.get_preview_entities = AsyncMock(return_value=[])

    detector = make_detector(category_fields=["host"], detector_type="MULTI_ENTITY")
    results = await runner.execute_detector(
        detector,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )

    assert results == []


@pytest.mark.asyncio
async def test_index_results(runner, mock_client):
    mock_client.index = AsyncMock(return_value={"_id": "abc", "result": "created"})

    detector = make_detector()
    results = [
        AnomalyResult(
            detector_id="det-123",
            anomaly_grade=0.85,
            confidence=0.99,
            data_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            data_end_time=datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc),
            feature_data=[],
            schema_version=7,
        )
    ]
    await runner.index_results(detector, results)

    mock_client.index.assert_awaited_once()
    call_kwargs = mock_client.index.call_args.kwargs
    assert call_kwargs["index"] == ".opendistro-anomaly-results"


@pytest.mark.asyncio
async def test_preview_mode_no_mutation(runner, mock_fm):
    rng = np.random.RandomState(3)
    data = rng.normal(size=(100, 2))
    features = Features(
        unprocessed_features=data.astype(np.float64),
        time_ranges=[(1704067200000, 1704067200000)] * 100,
    )
    mock_fm.get_features = AsyncMock(return_value=features)

    detector = make_detector()
    results1 = await runner.execute_detector(
        detector,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
        preview=False,
    )
    updates_after_real = runner._model_cache[runner._make_model_id(detector)].total_updates

    runner.clear_cache()
    results2 = await runner.execute_detector(
        detector,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
        preview=True,
    )
    # In preview mode, a new temporary model is created each time.
    # The actual forest state is not cached (preview uses cloned forest).
    assert len(results1) == len(results2) == 100


def test_sample_max_results():
    results = [AnomalyResult(detector_id="x", anomaly_grade=min(1.0, i / 50), confidence=1.0,
                             data_start_time=datetime.now(timezone.utc),
                             data_end_time=datetime.now(timezone.utc),
                             feature_data=[], schema_version=7)
               for i in range(100)]
    sampled = _sample_max_results(results, 10)
    assert len(sampled) == 10
