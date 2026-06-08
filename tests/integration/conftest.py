"""Shared fixtures for integration tests against Elasticsearch via testcontainers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import numpy as np
import pytest
import pytest_asyncio
from elasticsearch import AsyncElasticsearch
from testcontainers.elasticsearch import ElasticSearchContainer


def _make_doc(
    timestamp: datetime,
    cpu: float,
    memory: float,
    host: str = "server1",
) -> dict:
    return {
        "@timestamp": timestamp.isoformat(),
        "cpu": cpu,
        "memory": memory,
        "host": host,
    }


@pytest_asyncio.fixture
async def es_client(es_container: dict) -> AsyncGenerator[AsyncElasticsearch, None]:
    """
    Fresh ES client per test (required because aiohttp sessions bind to a
    specific event loop, and pytest-asyncio creates a new loop per test
    by default).
    """
    client = AsyncElasticsearch([es_container["url"]])
    yield client
    await client.close()


@pytest.fixture(scope="session")
def es_container() -> dict:
    """
    Shared ES container.  Returns connection info as a plain dict so the
    async client can be recreated per-test with the correct event loop.
    """
    container = ElasticSearchContainer("elasticsearch:8.15.3")
    container.start()
    try:
        url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(9200)}"
        yield {"url": url, "container": container}
    finally:
        container.stop()


@pytest_asyncio.fixture
async def sample_index(
    es_client: AsyncElasticsearch,
) -> AsyncGenerator[str, None]:
    """Create and populate a sample index with normal + anomalous data."""
    index = "test-metrics"
    await es_client.indices.create(
        index=index,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "cpu": {"type": "double"},
                    "memory": {"type": "double"},
                    "host": {"type": "keyword"},
                }
            },
        },
    )

    now = datetime.now(timezone.utc)
    rng = np.random.RandomState(42)
    docs: list[dict] = []

    # Normal data for 48 hours, 10-minute intervals
    for i in range(48 * 6):
        ts = now - timedelta(minutes=10 * (48 * 6 - i))
        cpu = float(rng.normal(loc=30.0, scale=2.0))
        mem = float(rng.normal(loc=60.0, scale=3.0))
        docs.append(_make_doc(ts, cpu, mem))

        if i % 6 == 0:
            docs.append(_make_doc(ts, cpu + 5.0, mem + 5.0, host="server2"))

    # Inject anomalies in the last 2 hours
    for i in range(6, 18):
        ts = now - timedelta(minutes=10 * (18 - i))
        cpu = float(rng.normal(loc=95.0, scale=1.0))
        mem = float(rng.normal(loc=95.0, scale=1.0))
        docs.append(_make_doc(ts, cpu, mem))

    actions = []
    for doc in docs:
        actions.append({"index": {"_index": index}})
        actions.append(doc)

    await es_client.bulk(operations=actions, refresh="wait_for")
    await es_client.indices.refresh(index=index)

    yield index

    await es_client.indices.delete(index=index, ignore_unavailable=True)


@pytest_asyncio.fixture
async def multi_entity_index(
    es_client: AsyncElasticsearch,
) -> AsyncGenerator[str, None]:
    """Create and populate a sample index with multi-entity data."""
    index = "test-multi-metrics"
    await es_client.indices.create(
        index=index,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "cpu": {"type": "double"},
                    "memory": {"type": "double"},
                    "host": {"type": "keyword"},
                }
            },
        },
    )

    now = datetime.now(timezone.utc)
    rng = np.random.RandomState(7)
    docs: list[dict] = []

    for host in ("web1", "web2", "web3"):
        base_cpu = float(rng.randint(20, 60))
        base_mem = float(rng.randint(40, 70))
        for i in range(48 * 6):
            ts = now - timedelta(minutes=10 * (48 * 6 - i))
            cpu = float(rng.normal(loc=base_cpu, scale=2.0))
            mem = float(rng.normal(loc=base_mem, scale=3.0))
            docs.append(_make_doc(ts, cpu, mem, host=host))

    actions = []
    for doc in docs:
        actions.append({"index": {"_index": index}})
        actions.append(doc)

    await es_client.bulk(operations=actions, refresh="wait_for")
    await es_client.indices.refresh(index=index)

    yield index

    await es_client.indices.delete(index=index, ignore_unavailable=True)
