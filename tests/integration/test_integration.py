"""Integration tests against a real Elasticsearch cluster via testcontainers."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from elasticsearch import AsyncElasticsearch

from src.client import ensure_index_exists
from src.feature_manager import FeatureManager
from src.models import (
    AnomalyDetector,
    Feature,
    IntervalTimeConfiguration,
)
from src.runner import AnomalyDetectorRunner


# ------------------------------------------------------------------ #
# Smoke tests
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_es_connection(es_client: AsyncElasticsearch) -> None:
    info = await es_client.info()
    assert "version" in info
    assert info["version"]["number"] >= "8.15.0"


@pytest.mark.asyncio
async def test_ensure_index(es_client: AsyncElasticsearch) -> None:
    created = await ensure_index_exists(es_client, "test-ensure")
    assert created is True
    created_again = await ensure_index_exists(es_client, "test-ensure")
    assert created_again is False


# ------------------------------------------------------------------ #
# FeatureManager integration
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_feature_manager_single_entity(
    es_client: AsyncElasticsearch, sample_index: str
) -> None:
    detector = AnomalyDetector(
        name="integration-test",
        time_field="@timestamp",
        indices=[sample_index],
        features=[
            Feature(name="cpu", aggregation={"avg": {"field": "cpu"}}),
            Feature(name="memory", aggregation={"avg": {"field": "memory"}}),
        ],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
    )

    fm = FeatureManager(client=es_client)
    start = datetime.now(timezone.utc) - timedelta(hours=8)
    end = datetime.now(timezone.utc)

    features = await fm.get_features(detector, start, end)

    assert features.num_samples > 0
    assert features.num_features == 2
    assert features.time_ranges


@pytest.mark.asyncio
async def test_feature_manager_multi_entity(
    es_client: AsyncElasticsearch, multi_entity_index: str
) -> None:
    detector = AnomalyDetector(
        name="integration-multi",
        time_field="@timestamp",
        indices=[multi_entity_index],
        features=[
            Feature(name="cpu", aggregation={"avg": {"field": "cpu"}}),
        ],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
        category_fields=["host"],
    )

    fm = FeatureManager(client=es_client)
    start = datetime.now(timezone.utc) - timedelta(hours=8)
    end = datetime.now(timezone.utc)

    entities = await fm.get_preview_entities(detector, start, end)
    assert len(entities) >= 3

    features = await fm.get_features_for_entity(
        detector, entities[0], start, end
    )
    assert features.num_samples > 0
    assert features.entity == entities[0]


# ------------------------------------------------------------------ #
# Runner integration (single entity)
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_runner_single_entity(
    es_client: AsyncElasticsearch, sample_index: str
) -> None:
    detector = AnomalyDetector(
        name="integration-runner",
        time_field="@timestamp",
        indices=[sample_index],
        features=[
            Feature(name="cpu", aggregation={"avg": {"field": "cpu"}}),
            Feature(name="memory", aggregation={"avg": {"field": "memory"}}),
        ],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
        result_index=".opendistro-anomaly-results",
    )

    fm = FeatureManager(client=es_client)
    runner = AnomalyDetectorRunner(
        client=es_client,
        feature_manager=fm,
    )

    start = datetime.now(timezone.utc) - timedelta(hours=4)
    end = datetime.now(timezone.utc)

    results = await runner.execute_detector(detector, start, end, preview=True)

    assert len(results) > 0
    # Most should have grades between 0 and 1
    for r in results:
        assert 0.0 <= r.anomaly_grade <= 1.0
        assert r.detector_id == detector.detector_id or detector.detector_id is None


@pytest.mark.asyncio
async def test_runner_multi_entity(
    es_client: AsyncElasticsearch, multi_entity_index: str
) -> None:
    detector = AnomalyDetector(
        name="integration-runner-multi",
        time_field="@timestamp",
        indices=[multi_entity_index],
        features=[
            Feature(name="cpu", aggregation={"avg": {"field": "cpu"}}),
        ],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
        category_fields=["host"],
        detector_type="MULTI_ENTITY",
        result_index=".opendistro-anomaly-results",
    )

    fm = FeatureManager(client=es_client)
    runner = AnomalyDetectorRunner(
        client=es_client,
        feature_manager=fm,
    )

    start = datetime.now(timezone.utc) - timedelta(hours=4)
    end = datetime.now(timezone.utc)

    results = await runner.execute_detector(detector, start, end, preview=True)

    assert len(results) > 0
    # Results should come from multiple hosts
    hosts = set()
    for r in results:
        if r.entity:
            hosts.add(r.entity.attributes.get("host"))
    assert len(hosts) >= 2


@pytest.mark.asyncio
async def test_runner_index_results(
    es_client: AsyncElasticsearch, sample_index: str
) -> None:
    detector = AnomalyDetector(
        name="integration-index",
        time_field="@timestamp",
        indices=[sample_index],
        features=[
            Feature(name="cpu", aggregation={"avg": {"field": "cpu"}}),
        ],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
        result_index=".opendistro-anomaly-results",
    )

    fm = FeatureManager(client=es_client)
    runner = AnomalyDetectorRunner(
        client=es_client,
        feature_manager=fm,
    )

    start = datetime.now(timezone.utc) - timedelta(hours=1)
    end = datetime.now(timezone.utc)

    results = await runner.execute_detector(detector, start, end, preview=True)
    assert len(results) > 0

    await runner.index_results(detector, results)

    # Give ES a moment to make docs searchable (default refresh interval is 1s)
    import asyncio
    await asyncio.sleep(1.5)

    # Verify results are in ES
    resp = await es_client.search(
        index=".opendistro-anomaly-results",
        body={"query": {"match_all": {}}},
    )
    assert resp["hits"]["total"]["value"] >= len(results)


# ------------------------------------------------------------------ #
# Cold-start training threshold
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_cold_start_triggers_training(
    es_client: AsyncElasticsearch, sample_index: str
) -> None:
    detector = AnomalyDetector(
        name="integration-cold-start",
        time_field="@timestamp",
        indices=[sample_index],
        features=[
            Feature(name="cpu", aggregation={"avg": {"field": "cpu"}}),
        ],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
    )

    fm = FeatureManager(client=es_client)
    runner = AnomalyDetectorRunner(
        client=es_client,
        feature_manager=fm,
    )

    # Use a large window to ensure enough samples for cold start
    start = datetime.now(timezone.utc) - timedelta(hours=24)
    end = datetime.now(timezone.utc)

    results = await runner.execute_detector(detector, start, end)

    # With enough training data, most results should be from an output-ready model
    ready_results = [r for r in results if r.anomaly_score is not None]
    assert len(ready_results) > 0
