"""Anomaly detection runner: orchestrates feature queries, scoring, and results."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from elasticsearch import AsyncElasticsearch

from src.client import index_document
from src.feature_manager import FeatureManager, Features
from src.models import (
    AnomalyDetector,
    AnomalyResult,
    DataByFeatureId,
    Entity,
    FeatureData,
    ThresholdingResult,
)
from src.rcf_wrapper import AnomalyDescriptor, RCFModel

logger = logging.getLogger(__name__)

MAX_PREVIEW_RESULTS = 50  # Java default


class AnomalyDetectorRunner:
    """
    Python analogue of Java ``AnomalyDetectorRunner`` + ``ADModelManager``.

    Orchestrates feature fetching, model training/scoring, result
    construction, and indexing for both single-entity and multi-entity
    detectors.
    """

    def __init__(
        self,
        client: AsyncElasticsearch,
        feature_manager: FeatureManager,
        max_preview_results: int = MAX_PREVIEW_RESULTS,
    ):
        self.client = client
        self.feature_manager = feature_manager
        self.max_preview_results = max_preview_results
        self._model_cache: Dict[str, RCFModel] = {}

    def _make_model_id(
        self, detector: AnomalyDetector, entity: Optional[Entity] = None
    ) -> str:
        if entity is None or not entity.attributes:
            return f"{detector.detector_id}"
        # Simple deterministic hash of entity attributes
        attr_str = "_".join(f"{k}={v}" for k, v in sorted(entity.attributes.items()))
        return f"{detector.detector_id}-{attr_str}"

    def _get_or_create_model(
        self,
        detector: AnomalyDetector,
        entity: Optional[Entity] = None,
    ) -> RCFModel:
        model_id = self._make_model_id(detector, entity)
        if model_id not in self._model_cache:
            self._model_cache[model_id] = RCFModel(detector, entity=entity)
            logger.info("Created cached RCF model %s", model_id)
        return self._model_cache[model_id]

    def clear_cache(self) -> None:
        """Drop all in-memory models (useful for testing)."""
        self._model_cache.clear()

    # ------------------------------------------------------------------ #
    # Core execution
    # ------------------------------------------------------------------ #

    async def execute_detector(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
        *,
        preview: bool = False,
    ) -> List[AnomalyResult]:
        """
        Run anomaly detection for the given detector and time window.

        Parameters
        ----------
        detector:   Detector configuration.
        start_time: Start of detection window (UTC).
        end_time:   End of detection window (UTC).
        preview:    If ``True``, do not mutate persisted checkpoints.

        Returns
        -------
        List of ``AnomalyResult`` objects.
        """
        execution_start = datetime.now(timezone.utc)

        if detector.is_multi_entity:
            results = await self._execute_multi_entity(
                detector, start_time, end_time, preview=preview
            )
        else:
            results = await self._execute_single_entity(
                detector, start_time, end_time, preview=preview
            )

        execution_end = datetime.now(timezone.utc)
        for r in results:
            r.execution_start_time = execution_start
            r.execution_end_time = execution_end

        return results

    # ------------------------------------------------------------------ #
    # Single entity
    # ------------------------------------------------------------------ #

    async def _execute_single_entity(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
        *,
        preview: bool = False,
    ) -> List[AnomalyResult]:
        features = await self.feature_manager.get_features(
            detector, start_time, end_time
        )
        return self._score_features(detector, features, preview=preview)

    # ------------------------------------------------------------------ #
    # Multi entity
    # ------------------------------------------------------------------ #

    async def _execute_multi_entity(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
        *,
        preview: bool = False,
    ) -> List[AnomalyResult]:
        entities = await self.feature_manager.get_preview_entities(
            detector, start_time, end_time
        )
        if not entities:
            logger.warning(
                "No entities found for multi-entity detector %s", detector.detector_id
            )
            return []

        all_results: List[AnomalyResult] = []
        # Process entities concurrently (up to 10 at a time)
        semaphore = asyncio.Semaphore(10)

        async def _process_entity(entity: Entity) -> List[AnomalyResult]:
            async with semaphore:
                features = await self.feature_manager.get_features_for_entity(
                    detector, entity, start_time, end_time
                )
                results = self._score_features(
                    detector, features, preview=preview
                )
                # Cap per-entity results for preview
                if preview and len(results) > self.max_preview_results:
                    results = _sample_max_results(results, self.max_preview_results)
                return results

        tasks = [_process_entity(e) for e in entities]
        for results in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(results, BaseException):
                logger.error("Entity processing failed: %s", results)
                continue
            all_results.extend(results)

        return all_results

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _score_features(
        self,
        detector: AnomalyDetector,
        features: Features,
        *,
        preview: bool = False,
    ) -> List[AnomalyResult]:
        if features.num_samples == 0:
            return []

        model = self._get_or_create_model(detector, features.entity)

        # Train thresholding model on cold-start window if necessary
        if not model.is_output_ready():
            logger.info(
                "Training cold-start model for %s with %d samples",
                self._make_model_id(detector, features.entity),
                features.num_samples,
            )
            model.train(features.unprocessed_features)

        # Score the batch
        if preview:
            descriptors = model.get_preview_results(features.unprocessed_features)
        else:
            descriptors = model.process_batch(features.unprocessed_features)

        return self._build_results(detector, features, descriptors)

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #

    def _build_results(
        self,
        detector: AnomalyDetector,
        features: Features,
        descriptors: List[AnomalyDescriptor],
    ) -> List[AnomalyResult]:
        results: List[AnomalyResult] = []
        enabled_features = detector.enabled_features

        for idx, descriptor in enumerate(descriptors):
            if idx >= len(features.time_ranges):
                break
            start_ms, end_ms = features.time_ranges[idx]
            start = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
            # Derive end from detection interval if placeholder
            if end_ms == start_ms:
                end = _add_interval(
                    start,
                    detector.detection_interval,
                )
            else:
                end = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)

            feature_datas = []
            if descriptor.actual and len(descriptor.actual) == len(enabled_features):
                for j, feat in enumerate(enabled_features):
                    feature_datas.append(
                        FeatureData(
                            feature_id=feat.id,
                            feature_name=feat.name,
                            data=float(descriptor.actual[j]),
                        )
                    )
            elif idx < features.num_samples:
                for j, feat in enumerate(enabled_features):
                    feature_datas.append(
                        FeatureData(
                            feature_id=feat.id,
                            feature_name=feat.name,
                            data=float(features.unprocessed_features[idx, j]),
                        )
                    )

            result = AnomalyResult(
                detector_id=detector.detector_id or "",
                anomaly_score=descriptor.rcf_score,
                anomaly_grade=descriptor.anomaly_grade,
                confidence=descriptor.confidence,
                feature_data=feature_datas,
                data_start_time=start,
                data_end_time=end,
                entity=features.entity,
                schema_version=7,
                model_id=self._make_model_id(detector, features.entity),
            )
            if descriptor.relevant_attribution:
                result.relevant_attribution = [
                    DataByFeatureId(
                        feature_id=enabled_features[j].id if j < len(enabled_features) else None,
                        data=float(descriptor.relevant_attribution[j]),
                    )
                    for j in range(len(descriptor.relevant_attribution))
                ]
            results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # Result indexing
    # ------------------------------------------------------------------ #

    async def index_results(
        self,
        detector: AnomalyDetector,
        results: List[AnomalyResult],
    ) -> None:
        """Write results to the detector's configured result index."""
        if not results:
            return
        index = detector.result_index or ".opendistro-anomaly-results"
        for result in results:
            await index_document(
                self.client,
                index,
                result.model_dump(by_alias=True),
            )
        logger.info("Indexed %d results to %s", len(results), index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_interval(
    dt: datetime, interval: "IntervalTimeConfiguration"
) -> datetime:
    unit = interval.unit.lower()
    if unit in ("s", "sec", "secs", "second", "seconds"):
        return dt.replace(second=dt.second + interval.interval)
    if unit in ("m", "min", "mins", "minute", "minutes"):
        return dt.replace(minute=dt.minute + interval.interval)
    if unit in ("h", "hr", "hrs", "hour", "hours"):
        return dt.replace(hour=dt.hour + interval.interval)
    if unit in ("d", "day", "days"):
        return dt.replace(day=dt.day + interval.interval)
    raise ValueError(f"Unsupported interval unit: {unit}")


def _sample_max_results(
    results: List[AnomalyResult], max_results: int
) -> List[AnomalyResult]:
    """Uniformly sample at most ``max_results`` from the list."""
    if len(results) <= max_results:
        return results
    indices = np.linspace(0, len(results) - 1, max_results, dtype=int)
    return [results[i] for i in indices]
