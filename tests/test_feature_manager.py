import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.feature_manager import FeatureManager, Features, _forward_fill_1d, _linear_interpolate_1d, _extract_agg_value
from src.models import AnomalyDetector, Feature, IntervalTimeConfiguration, Entity, ImputationOption, ImputationMethod


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
        shingle_size=8,
    )
    defaults.update(kwargs)
    return AnomalyDetector(**defaults)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def fm(mock_client):
    return FeatureManager(client=mock_client)


# ------------------------------------------------------------------ #
# Query construction
# ------------------------------------------------------------------ #


def test_build_single_entity_request(fm):
    detector = make_detector()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    body = fm._build_single_entity_request(detector, start, end)

    assert "query" in body
    assert "aggs" in body
    assert body["size"] == 0
    assert body["track_total_hits"] is False

    buckets = body["aggs"]["buckets"]["date_histogram"]
    assert buckets["field"] == "@timestamp"
    assert buckets["fixed_interval"] == "10m"
    assert buckets["min_doc_count"] == 1

    sub_aggs = body["aggs"]["buckets"]["aggs"]
    assert "feature_0" in sub_aggs
    assert "feature_1" in sub_aggs
    assert sub_aggs["feature_0"]["avg"]["field"] == "cpu"


def test_build_single_entity_request_with_filter(fm):
    detector = make_detector(
        filter_query={"term": {"host": "server1"}}
    )
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    body = fm._build_single_entity_request(detector, start, end)

    filters = body["query"]["bool"]["filter"]
    assert any("range" in f for f in filters)
    assert any(f == {"term": {"host": "server1"}} for f in filters)


def test_build_entity_request(fm):
    detector = make_detector(category_fields=["host"])
    entity = Entity(attributes={"host": "server1"})
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    body = fm._build_entity_request(detector, entity, start, end)

    filters = body["query"]["bool"]["filter"]
    assert any(f == {"term": {"host": "server1"}} for f in filters)


# ------------------------------------------------------------------ #
# Parse buckets
# ------------------------------------------------------------------ #


def test_parse_buckets_empty():
    resp = {"aggregations": {"buckets": {"buckets": []}}}
    vectors, ranges = FeatureManager._parse_buckets(resp, 2)
    assert vectors.shape == (0, 2)
    assert ranges == []


def test_parse_buckets_with_values():
    resp = {
        "aggregations": {
            "buckets": {
                "buckets": [
                    {
                        "key": 1704067200000,
                        "doc_count": 10,
                        "feature_0": {"value": 12.3},
                        "feature_1": {"value": 45.6},
                    },
                    {
                        "key": 1704067800000,
                        "doc_count": 8,
                        "feature_0": {"value": 13.0},
                        "feature_1": {"value": 46.0},
                    },
                ]
            }
        }
    }
    vectors, ranges = FeatureManager._parse_buckets(resp, 2)
    assert vectors.shape == (2, 2)
    np.testing.assert_array_equal(vectors, np.array([[12.3, 45.6], [13.0, 46.0]]))
    assert ranges == [(1704067200000, 1704067200000), (1704067800000, 1704067800000)]


def test_parse_buckets_with_missing():
    resp = {
        "aggregations": {
            "buckets": {
                "buckets": [
                    {
                        "key": 1704067200000,
                        "doc_count": 10,
                        "feature_0": {"value": 12.3},
                    },
                ]
            }
        }
    }
    vectors, ranges = FeatureManager._parse_buckets(resp, 2)
    assert vectors.shape == (1, 2)
    assert vectors[0, 0] == 12.3
    assert np.isnan(vectors[0, 1])


# ------------------------------------------------------------------ #
# Imputation helpers
# ------------------------------------------------------------------ #


def test_forward_fill_1d():
    arr = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float64)
    _forward_fill_1d(arr)
    np.testing.assert_array_equal(arr, np.array([1.0, 1.0, 1.0, 3.0, 3.0]))


def test_forward_fill_1d_all_nan():
    arr = np.array([np.nan, np.nan], dtype=np.float64)
    _forward_fill_1d(arr)
    assert np.all(np.isnan(arr))


def test_linear_interpolate_1d():
    arr = np.array([10.0, np.nan, np.nan, 40.0], dtype=np.float64)
    _linear_interpolate_1d(arr)
    np.testing.assert_array_almost_equal(arr, np.array([10.0, 20.0, 30.0, 40.0]))


def test_linear_interpolate_1d_single_valid():
    arr = np.array([np.nan, 5.0, np.nan], dtype=np.float64)
    _linear_interpolate_1d(arr)
    np.testing.assert_array_equal(arr, np.array([5.0, 5.0, 5.0]))


def test_impute_zero():
    data = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float64)
    result, mask = FeatureManager.impute(data, ImputationOption(method=ImputationMethod.ZERO))
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0
    assert mask is not None
    assert mask[0, 1]


def test_impute_linear():
    # 3 time steps, 1 feature — middle value should interpolate
    data = np.array([[10.0], [np.nan], [30.0]], dtype=np.float64)
    result, mask = FeatureManager.impute(data, ImputationOption(method=ImputationMethod.LINEAR))
    assert result[1, 0] == 20.0


def test_impute_previous():
    # 4 time steps, 1 feature
    data = np.array([[np.nan], [10.0], [np.nan], [30.0]], dtype=np.float64)
    result, mask = FeatureManager.impute(data, ImputationOption(method=ImputationMethod.PREVIOUS))
    assert result[0, 0] == 10.0  # back-filled from first valid
    assert result[2, 0] == 10.0  # forward-filled


def test_impute_fixed_values():
    data = np.array([[1.0, np.nan]], dtype=np.float64)
    option = ImputationOption(method=ImputationMethod.FIXED_VALUES)
    result, mask = FeatureManager.impute(
        data, option, feature_names=["cpu", "mem"], default_fills={"mem": 99.0}
    )
    assert result[0, 1] == 99.0


# ------------------------------------------------------------------ #
# FeatureManager async methods
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_features(fm, mock_client):
    mock_client.search = AsyncMock(
        return_value={
            "aggregations": {
                "buckets": {
                    "buckets": [
                        {"key": 1704067200000, "doc_count": 5, "feature_0": {"value": 10.0}, "feature_1": {"value": 20.0}},
                        {"key": 1704067800000, "doc_count": 5, "feature_0": {"value": 11.0}, "feature_1": {"value": 21.0}},
                    ]
                }
            }
        }
    )
    detector = make_detector()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    features = await fm.get_features(detector, start, end)

    assert features.num_samples == 2
    assert features.num_features == 2
    np.testing.assert_array_equal(
        features.unprocessed_features,
        np.array([[10.0, 20.0], [11.0, 21.0]]),
    )


@pytest.mark.asyncio
async def test_get_features_for_entity(fm, mock_client):
    mock_client.search = AsyncMock(
        return_value={
            "aggregations": {
                "buckets": {
                    "buckets": [
                        {"key": 1704067200000, "doc_count": 1, "feature_0": {"value": 5.0}},
                    ]
                }
            }
        }
    )
    detector = make_detector(category_fields=["host"])
    entity = Entity(attributes={"host": "web1"})
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    features = await fm.get_features_for_entity(detector, entity, start, end)

    assert features.num_samples == 1
    assert features.entity == entity


@pytest.mark.asyncio
async def test_get_preview_entities(fm, mock_client):
    mock_client.search = AsyncMock(
        return_value={
            "aggregations": {
                "top_entities": {
                    "buckets": [
                        {"key": {"host": "web1"}, "doc_count": 100},
                        {"key": {"host": "web2"}, "doc_count": 50},
                    ]
                }
            }
        }
    )
    detector = make_detector(category_fields=["host"])
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    entities = await fm.get_preview_entities(detector, start, end)

    assert len(entities) == 2
    assert entities[0].attributes == {"host": "web1"}
    assert entities[1].attributes == {"host": "web2"}


# ------------------------------------------------------------------ #
# Features dataclass helpers
# ------------------------------------------------------------------ #


def test_features_build_shingles():
    raw = np.arange(24).reshape(8, 3).astype(np.float64)
    f = Features(unprocessed_features=raw, time_ranges=[])
    shingles = f.build_shingles(shingle_size=4)
    assert shingles.shape == (5, 12)  # 8 - 4 + 1 = 5, 3 * 4 = 12


def test_features_build_shingles_too_few_samples():
    raw = np.arange(6).reshape(2, 3).astype(np.float64)
    f = Features(unprocessed_features=raw, time_ranges=[])
    shingles = f.build_shingles(shingle_size=4)
    assert shingles.shape == (0, 12)


# ------------------------------------------------------------------ #
# _extract_agg_value edge cases
# ------------------------------------------------------------------ #


def test_extract_agg_value_value():
    assert _extract_agg_value({"value": 42.0}) == 42.0


def test_extract_agg_value_none():
    assert np.isnan(_extract_agg_value({"value": None}))


def test_extract_agg_value_percentile():
    agg = {"values": {"50.0": 100.0, "75.0": 150.0, "95.0": 200.0}}
    assert _extract_agg_value(agg) == 150.0


def test_extract_agg_value_unknown():
    assert np.isnan(_extract_agg_value({"doc_count": 5}))
