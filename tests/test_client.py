import pytest
from unittest.mock import AsyncMock, MagicMock

from src.client import (
    create_es_client,
    retry_with_backoff,
    search,
    index_document,
    ensure_index_exists,
    composite_search,
    CHECKPOINT_INDEX_NAME,
    CONFIG_INDEX,
)


def test_create_es_client_returns_instance():
    """Factory should return an AsyncElasticsearch-like object."""
    client = create_es_client(["http://localhost:9200"], username="elastic", password="changeme")
    assert client is not None


@pytest.mark.asyncio
async def test_retry_with_backoff_succeeds_first_try():
    @retry_with_backoff(max_retries=2)
    async def flaky() -> str:
        return "ok"

    assert await flaky() == "ok"


@pytest.mark.asyncio
async def test_retry_with_backoff_retries_on_transport_error():
    from elasticsearch.exceptions import ConnectionError

    call_count = 0

    @retry_with_backoff(max_retries=2, initial_wait=0.01)
    async def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("boom")
        return "ok"

    assert await flaky() == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_exhaustion():
    from elasticsearch.exceptions import ConnectionError

    @retry_with_backoff(max_retries=1, initial_wait=0.01)
    async def always_fails() -> str:
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        await always_fails()


@pytest.mark.asyncio
async def test_search_wraps_client():
    mock_client = MagicMock()
    mock_client.search = AsyncMock(return_value={"hits": {"total": {"value": 0}}})

    resp = await search(mock_client, "test-index", {"match_all": {}})
    assert resp["hits"]["total"]["value"] == 0
    mock_client.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_index_document_wraps_client():
    mock_client = MagicMock()
    mock_client.index = AsyncMock(return_value={"_id": "abc", "result": "created"})

    resp = await index_document(mock_client, "test-index", {"foo": "bar"}, doc_id="abc")
    assert resp["result"] == "created"
    mock_client.index.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_index_exists_creates_when_missing():
    mock_client = MagicMock()
    mock_client.indices = MagicMock()
    mock_client.indices.exists = AsyncMock(return_value=False)
    mock_client.indices.create = AsyncMock(return_value={"acknowledged": True})

    created = await ensure_index_exists(mock_client, "new-index")
    assert created is True
    mock_client.indices.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_index_exists_skips_when_present():
    mock_client = MagicMock()
    mock_client.indices = MagicMock()
    mock_client.indices.exists = AsyncMock(return_value=True)

    created = await ensure_index_exists(mock_client, "existing-index")
    assert created is False


@pytest.mark.asyncio
async def test_composite_search_paginates():
    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        side_effect=[
            {
                "aggregations": {
                    "buckets": {
                        "buckets": [{"key": {"host": "a"}, "doc_count": 5}],
                        "after_key": {"host": "a"},
                    }
                }
            },
            {
                "aggregations": {
                    "buckets": {
                        "buckets": [{"key": {"host": "b"}, "doc_count": 3}],
                    }
                }
            },
        ]
    )

    buckets = await composite_search(
        mock_client, "test-index", {"sources": [{"host": {"terms": {"field": "host"}}}]}
    )
    assert len(buckets) == 2
