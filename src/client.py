"""Async Elasticsearch client factory and index management helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, TypeVar

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, ConnectionTimeout, TransportError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index / alias names (mirroring Java ADCommonName constants)
# ---------------------------------------------------------------------------

CHECKPOINT_INDEX_NAME = ".opendistro-anomaly-checkpoints"
DETECTION_STATE_INDEX = ".opendistro-anomaly-detection-state"
CONFIG_INDEX = ".opendistro-anomaly-detectors"
ANOMALY_RESULT_INDEX_ALIAS = ".opendistro-anomaly-results"
CUSTOM_RESULT_INDEX_PREFIX = "opensearch-ad-plugin-result-"

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def create_es_client(
    hosts: List[str],
    *,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    verify_certs: bool = True,
    ca_certs: Optional[str] = None,
    max_retries: int = 3,
    retry_on_timeout: bool = True,
    request_timeout: float = 30.0,
    max_connections: int = 10,
    **kwargs: Any,
) -> AsyncElasticsearch:
    """
    Create an :class:`AsyncElasticsearch` client with sensible defaults for
    anomaly-detection workloads.

    Parameters
    ----------
    hosts:            Elasticsearch node URLs, e.g. ["http://localhost:9200"]
    username:         Basic-auth username (optional).
    password:         Basic-auth password (optional).
    api_key:          Base64-encoded API key (optional; ``username`` gets ignored).
    verify_certs:     Whether to verify TLS certificates.
    ca_certs:         Path to CA bundle for TLS verification.
    max_retries:      Number of retries on retriable transport errors.
    retry_on_timeout: Retry when a read / connection timeout occurs.
    request_timeout:  Default request timeout in seconds.
    max_connections:  Max simultaneous HTTP connections per node.
    """
    auth_kwargs: Dict[str, Any] = {}
    if api_key:
        auth_kwargs["api_key"] = api_key
    elif username and password:
        auth_kwargs["basic_auth"] = (username, password)

    client = AsyncElasticsearch(
        hosts=hosts,
        verify_certs=verify_certs,
        ca_certs=ca_certs,
        max_retries=max_retries,
        retry_on_timeout=retry_on_timeout,
        request_timeout=request_timeout,
        connections_per_node=max_connections,
        **auth_kwargs,
        **kwargs,
    )
    return client


# ---------------------------------------------------------------------------
# Exponential-backoff retry decorator
# ---------------------------------------------------------------------------

T = TypeVar("T")

RETRIABLE_EXCEPTIONS = (ConnectionError, ConnectionTimeout, TransportError)


def retry_with_backoff(
    *,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    backoff_multiplier: float = 2.0,
    retriable_exceptions: tuple = RETRIABLE_EXCEPTIONS,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator that retries an ``async`` function on retriable exceptions with
    exponential backoff.

    Example::

        @retry_with_backoff(max_retries=5)
        async def fetch_features(...) -> ...
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            wait = initial_wait
            last_exc: Optional[BaseException] = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retriable_exceptions as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        logger.error(
                            "%s exhausted all %d retries. Last error: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
                        raise

                    # TransportError might wrap a non-retriable HTTP status;
                    # retry only on 429 / 5xx / timeout-like errors.
                    if isinstance(exc, TransportError) and hasattr(exc, "status_code"):
                        status = exc.status_code  # type: ignore[attr-defined]
                        if status is not None and status < 500 and status != 429:
                            raise

                    logger.warning(
                        "%s attempt %d/%d failed (%s). Retrying in %.1fs...",
                        func.__name__,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    wait = min(wait * backoff_multiplier, max_wait)

            # Should never reach here, but satisfy type-checker.
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Async helpers (search, index, bulk)
# ---------------------------------------------------------------------------

@retry_with_backoff()
async def search(
    client: AsyncElasticsearch,
    index: str,
    query: Dict[str, Any],
    *,
    size: int = 0,
    aggs: Optional[Dict[str, Any]] = None,
    scroll: Optional[str] = None,
    track_total_hits: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Async search wrapper with built-in retry/backoff."""
    body: Dict[str, Any] = {
        "query": query,
        "track_total_hits": track_total_hits,
        "size": size,
    }
    if aggs:
        body["aggs"] = aggs
    if scroll:
        body["scroll"] = scroll
    resp = await client.search(index=index, body=body, **kwargs)
    return resp  # type: ignore[return-value]


@retry_with_backoff()
async def index_document(
    client: AsyncElasticsearch,
    index: str,
    document: Dict[str, Any],
    *,
    doc_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Index a single document with retry/backoff."""
    resp = await client.index(index=index, id=doc_id, document=document, **kwargs)
    return resp  # type: ignore[return-value]


@retry_with_backoff()
async def bulk_index(
    client: AsyncElasticsearch,
    actions: List[Dict[str, Any]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Bulk-index operations with retry/backoff."""
    resp = await client.bulk(operations=actions, **kwargs)
    return resp  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Index management helpers
# ---------------------------------------------------------------------------

async def ensure_index_exists(
    client: AsyncElasticsearch,
    index: str,
    mapping: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> bool:
    """
    Create ``index`` if it does not already exist.

    Returns ``True`` if the index was created by this call, ``False`` if it
    already existed.
    """
    exists = await client.indices.exists(index=index)
    if exists:
        return False

    body: Dict[str, Any] = {}
    if settings:
        body["settings"] = settings
    if mapping:
        body["mappings"] = mapping

    await client.indices.create(index=index, body=body or None, **kwargs)
    logger.info("Created index %s", index)
    return True


async def ensure_alias(
    client: AsyncElasticsearch,
    alias: str,
    index_pattern: str,
    **kwargs: Any,
) -> None:
    """Create an alias pointing to ``index_pattern`` if not already present."""
    aliases = await client.indices.get_alias(name=alias, ignore_unavailable=True)
    if alias not in (aliases or {}):
        await client.indices.put_alias(index=index_pattern, name=alias, **kwargs)
        logger.info("Created alias %s -> %s", alias, index_pattern)


# ---------------------------------------------------------------------------
# Composite aggregation pagination helper
# ---------------------------------------------------------------------------

@retry_with_backoff()
async def composite_search(
    client: AsyncElasticsearch,
    index: str,
    composite_agg: Dict[str, Any],
    query: Optional[Dict[str, Any]] = None,
    *,
    size: int = 0,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Execute a composite aggregation and return **all** buckets by paginating
    through ``after_key``.

    Returns a list of raw bucket dicts (``{"key": {...}, "doc_count": N}``).
    """
    buckets: List[Dict[str, Any]] = []
    after_key: Optional[Dict[str, Any]] = None

    while True:
        aggs: Dict[str, Any] = {
            "buckets": {
                "composite": {**composite_agg, "size": size or 1000},
            }
        }
        if after_key:
            aggs["buckets"]["composite"]["after"] = after_key

        resp = await client.search(
            index=index,
            body={
                "query": query or {"match_all": {}},
                "aggs": aggs,
                "size": 0,
            },
            **kwargs,
        )
        agg_resp = resp.get("aggregations", {}).get("buckets", {})  # type: ignore[attr-defined]
        current = agg_resp.get("buckets", [])
        buckets.extend(current)
        after_key = agg_resp.get("after_key")
        if not after_key or not current:
            break

    return buckets
