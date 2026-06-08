"""Checkpoint persistence: read/write model state to/from Elasticsearch."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from elasticsearch import AsyncElasticsearch

from src.client import (
    CHECKPOINT_INDEX_NAME,
    ensure_index_exists,
    retry_with_backoff,
    search,
)
from src.models import AnomalyDetector, Entity
from src.rcf_wrapper import RCFModel

logger = logging.getLogger(__name__)

CHECKPOINT_MAPPING: Dict[str, Any] = {
    "dynamic": True,
    "properties": {
        "detectorId": {"type": "keyword"},
        "modelId": {"type": "keyword"},
        "timestamp": {"type": "date", "format": "strict_date_time||epoch_millis"},
        "schema_version": {"type": "integer"},
        "entity": {
            "type": "nested",
            "properties": {
                "name": {"type": "keyword"},
                "value": {"type": "keyword"},
            },
        },
        "model_state": {"type": "object", "enabled": False},
    },
}


class CheckpointDao:
    """
    DAO for persisting and loading per-entity RCF + thresholding checkpoints
    to ``.opendistro-anomaly-checkpoints``.
    """

    def __init__(self, client: AsyncElasticsearch):
        self.client = client

    async def ensure_index(self) -> None:
        """Create the checkpoint index if it does not yet exist."""
        await ensure_index_exists(
            self.client,
            CHECKPOINT_INDEX_NAME,
            mapping=CHECKPOINT_MAPPING,
            settings={"number_of_shards": 1, "number_of_replicas": 0},
        )

    @retry_with_backoff()
    async def get_checkpoint(
        self,
        model_id: str,
        detector_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint document for a model ID."""
        query: Dict[str, Any] = {"term": {"modelId": model_id}}
        if detector_id:
            query = {
                "bool": {
                    "filter": [
                        {"term": {"modelId": model_id}},
                        {"term": {"detectorId": detector_id}},
                    ]
                }
            }

        resp = await search(
            self.client,
            CHECKPOINT_INDEX_NAME,
            query,
            size=1,
            sort=[{"timestamp": {"order": "desc"}}],
        )
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return None
        return hits[0]["_source"]

    @retry_with_backoff()
    async def put_checkpoint(
        self,
        model_id: str,
        detector_id: str,
        model_state: Dict[str, Any],
        entity: Optional[Entity] = None,
        schema_version: int = 5,
    ) -> None:
        """Persist a checkpoint document."""
        doc: Dict[str, Any] = {
            "detectorId": detector_id,
            "modelId": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": schema_version,
            "model_state": model_state,
        }
        if entity:
            doc["entity"] = entity.to_es_nested()

        await self.client.index(
            index=CHECKPOINT_INDEX_NAME,
            document=doc,
            refresh="wait_for",
        )
        logger.info("Saved checkpoint for model %s", model_id)

    # ------------------------------------------------------------------ #
    # High-level helpers
    # ------------------------------------------------------------------ #

    async def load_model(
        self,
        detector: AnomalyDetector,
        entity: Optional[Entity] = None,
        model_id: Optional[str] = None,
    ) -> Optional[RCFModel]:
        """Restore an ``RCFModel`` from the latest checkpoint, if any."""
        if model_id is None:
            entity_str = "-".join(f"{k}={v}" for k, v in sorted((entity or {}).attributes.items())) if entity else ""
            model_id = f"{detector.detector_id or 'unknown'}-{entity_str}" if entity_str else (detector.detector_id or "unknown")

        data = await self.get_checkpoint(model_id, detector.detector_id)
        if data is None:
            return None

        model_state = data.get("model_state")
        if model_state is None:
            return None

        try:
            return RCFModel.from_dict(model_state, detector)
        except Exception as exc:
            logger.error("Failed to restore checkpoint for %s: %s", model_id, exc)
            return None

    async def save_model(
        self,
        model: RCFModel,
        detector: AnomalyDetector,
        entity: Optional[Entity] = None,
    ) -> None:
        """Persist the current state of ``model``."""
        model_state = model.to_dict()
        model_id = model_state.get("detector_id", detector.detector_id or "unknown")
        await self.put_checkpoint(
            model_id=model_id,
            detector_id=detector.detector_id or "unknown",
            model_state=model_state,
            entity=entity or model.entity,
        )
