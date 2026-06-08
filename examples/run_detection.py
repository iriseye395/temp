"""
Example async anomaly detection runner using elasticsearch[async].

This script demonstrates:
1. Creating an AsyncElasticsearch client with auth / TLS.
2. Configuring an AnomalyDetector (single-entity or multi-entity).
3. Running detection over a time window.
4. Printing results with anomaly grades and scores.
5. Handling connection errors and timeouts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List

# Allow running this example without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, ConnectionTimeout

from src.client import create_es_client
from src.feature_manager import FeatureManager
from src.models import (
    AnomalyDetector,
    Feature,
    IntervalTimeConfiguration,
)
from src.runner import AnomalyDetectorRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async anomaly detection runner (elasticsearch[async])"
    )
    # Elasticsearch connection
    conn = parser.add_argument_group("Elasticsearch connection")
    conn.add_argument(
        "--host", default="http://localhost:9200", help="ES node URL"
    )
    conn.add_argument("--username", default=None, help="Basic-auth username")
    conn.add_argument("--password", default=None, help="Basic-auth password")
    conn.add_argument("--api-key", default=None, help="Base64 API key")
    conn.add_argument("--no-verify-certs", action="store_true", help="Skip TLS verify")
    conn.add_argument("--ca-certs", default=None, help="CA bundle path")
    conn.add_argument(
        "--request-timeout", type=float, default=30.0, help="ES request timeout"
    )
    conn.add_argument(
        "--max-retries", type=int, default=3, help="Max retries on retriable errors"
    )

    # Detector config
    det = parser.add_argument_group("Detector configuration")
    det.add_argument("--indices", required=True, help="Comma-separated source indices")
    det.add_argument("--time-field", default="@timestamp", help="Timestamp field name")
    det.add_argument(
        "--features", required=True, help='JSON array of Feature objects, e.g. "[{name:cpu,agg:{avg:{field:cpu}}}]"'
    )
    det.add_argument(
        "--interval-minutes", type=int, default=10, help="Detection interval in minutes"
    )
    det.add_argument("--shingle-size", type=int, default=8, help="Shingle size")
    det.add_argument(
        "--category-fields", default="", help="Comma-separated entity fields (enables multi-entity)"
    )
    det.add_argument(
        "--filter-query", default=None, help='Optional query DSL JSON filter'
    )

    # Time window
    win = parser.add_argument_group("Detection window")
    win.add_argument(
        "--start-time", default=None, help="ISO start time (defaults to 24h ago)"
    )
    win.add_argument(
        "--end-time", default=None, help="ISO end time (defaults to now)"
    )
    win.add_argument("--hours", type=int, default=24, help="Look-back hours if --start-time not given")

    # Behavior
    beh = parser.add_argument_group("Behavior")
    beh.add_argument(
        "--preview", action="store_true", help="Preview mode (no checkpoint mutation)"
    )
    beh.add_argument(
        "--index-results", action="store_true", help="Write results to ES result index"
    )
    beh.add_argument(
        "--max-preview-results", type=int, default=50, help="Max results per entity in preview"
    )

    return parser.parse_args(argv)


def build_detector(args: argparse.Namespace) -> AnomalyDetector:
    """Construct an AnomalyDetector from CLI args."""
    features_raw = json.loads(args.features)
    features = [Feature(**f) for f in features_raw]

    category_fields = [f.strip() for f in args.category_fields.split(",") if f.strip()]

    filter_query = None
    if args.filter_query:
        filter_query = json.loads(args.filter_query)

    return AnomalyDetector(
        name="cli-detector",
        time_field=args.time_field,
        indices=args.indices.split(","),
        features=features,
        detection_interval=IntervalTimeConfiguration(
            interval=args.interval_minutes, unit="minutes"
        ),
        shingle_size=args.shingle_size,
        category_fields=category_fields,
        filter_query=filter_query,
        result_index=".opendistro-anomaly-results",
    )


def parse_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    """Resolve detection window from CLI args."""
    if args.end_time:
        end = datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
    else:
        end = datetime.now(timezone.utc)

    if args.start_time:
        start = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
    else:
        start = end - timedelta(hours=args.hours)

    return start, end


async def run_detection(args: argparse.Namespace) -> None:
    """Main async entrypoint."""
    logger.info("Connecting to Elasticsearch at %s", args.host)

    client = create_es_client(
        hosts=[args.host],
        username=args.username,
        password=args.password,
        api_key=args.api_key,
        verify_certs=not args.no_verify_certs,
        ca_certs=args.ca_certs,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )

    try:
        # Smoke test
        info = await client.info()
        logger.info(
            "Connected: %s (cluster %s)",
            info.get("version", {}).get("number", "unknown"),
            info.get("cluster_name", "unknown"),
        )
    except (ConnectionError, ConnectionTimeout) as exc:
        logger.error("Failed to connect to Elasticsearch: %s", exc)
        sys.exit(1)

    detector = build_detector(args)
    start, end = parse_window(args)
    logger.info(
        "Running detector '%s' from %s to %s (preview=%s)",
        detector.name,
        start.isoformat(),
        end.isoformat(),
        args.preview,
    )

    feature_manager = FeatureManager(client)
    runner = AnomalyDetectorRunner(
        client=client,
        feature_manager=feature_manager,
        max_preview_results=args.max_preview_results,
    )

    try:
        results = await runner.execute_detector(
            detector, start, end, preview=args.preview
        )
    except Exception as exc:
        logger.error("Detection failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("Detection completed: %d results", len(results))

    # Print results
    for result in results:
        entity_info = ""
        if result.entity:
            entity_info = f" entity={result.entity.attributes}"
        logger.info(
            "  grade=%.3f score=%.3f confidence=%.3f start=%s%s",
            result.anomaly_grade,
            result.anomaly_score,
            result.confidence,
            result.data_start_time.isoformat() if result.data_start_time else "",
            entity_info,
        )

    if args.index_results:
        await runner.index_results(detector, results)
        logger.info("Results indexed to %s", detector.result_index)

    await client.close()
    logger.info("Done")


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run_detection(args))


if __name__ == "__main__":
    main()
