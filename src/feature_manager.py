"""Feature manager: fetch feature data from Elasticsearch, impute gaps, and build shingles."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from elasticsearch import AsyncElasticsearch

from src.models import AnomalyDetector, Entity, Feature, ImputationMethod, ImputationOption

logger = logging.getLogger(__name__)

DEFAULT_MAX_MISSING_PROPORTION = 0.25


@dataclass
class Features:
    """Container for extracted feature vectors and metadata."""

    # Shape: (time_steps, num_features)
    unprocessed_features: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0))
    )
    # List of (start_epoch_ms, end_epoch_ms) per time bucket
    time_ranges: List[Tuple[int, int]] = field(default_factory=list)
    # Which feature indices were imputed per time step (boolean mask)
    imputed_mask: Optional[np.ndarray] = field(default=None)
    # The entity these features belong to (None for single-entity)
    entity: Optional[Entity] = field(default=None)

    @property
    def num_samples(self) -> int:
        return self.unprocessed_features.shape[0]

    @property
    def num_features(self) -> int:
        return self.unprocessed_features.shape[1]

    def build_shingles(self, shingle_size: int) -> np.ndarray:
        """
        Convert raw feature vectors into shingled vectors.

        Returns an array of shape (num_samples - shingle_size + 1,
        num_features * shingle_size).
        """
        if self.num_samples < shingle_size:
            return np.zeros((0, self.num_features * shingle_size))
        n = self.num_samples - shingle_size + 1
        dim = self.num_features * shingle_size
        shingles = np.zeros((n, dim))
        for i in range(n):
            shingles[i] = self.unprocessed_features[i : i + shingle_size].ravel()
        return shingles


class FeatureManager:
    def __init__(
        self,
        client: AsyncElasticsearch,
        max_missing_proportion: float = DEFAULT_MAX_MISSING_PROPORTION,
    ):
        self.client = client
        self.max_missing_proportion = max_missing_proportion

    # ------------------------------------------------------------------ #
    # Query construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detection_interval_to_es(detector: AnomalyDetector) -> str:
        """Convert IntervalTimeConfiguration to Elasticsearch interval string."""
        interval = detector.detection_interval
        unit = interval.unit.lower()
        if unit in ("s", "sec", "secs", "second", "seconds"):
            return f"{interval.interval}s"
        if unit in ("m", "min", "mins", "minute", "minutes"):
            return f"{interval.interval}m"
        if unit in ("h", "hr", "hrs", "hour", "hours"):
            return f"{interval.interval}h"
        if unit in ("d", "day", "days"):
            return f"{interval.interval}d"
        raise ValueError(f"Unsupported time interval unit: {unit}")

    def _build_base_query(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Build the bool query with time range and optional detector filter."""
        time_query = {
            "range": {
                detector.time_field: {
                    "gte": int(start_time.timestamp() * 1000),
                    "lte": int(end_time.timestamp() * 1000),
                    "format": "epoch_millis",
                }
            }
        }
        must = [time_query]
        if detector.filter_query:
            must.append(detector.filter_query)
        return {"bool": {"filter": must}}

    def _build_feature_aggs(self, detector: AnomalyDetector) -> Dict[str, Any]:
        """Build per-feature sub-aggregations."""
        aggs: Dict[str, Any] = {}
        for idx, feat in enumerate(detector.enabled_features):
            aggs[f"feature_{idx}"] = feat.aggregation
        return aggs

    def _build_single_entity_request(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Build search body for single-entity feature aggregation."""
        interval_str = self._detection_interval_to_es(detector)
        aggs = self._build_feature_aggs(detector)
        return {
            "query": self._build_base_query(detector, start_time, end_time),
            "aggs": {
                "buckets": {
                    "date_histogram": {
                        "field": detector.time_field,
                        "fixed_interval": interval_str,
                        "min_doc_count": 1,
                    },
                    "aggs": aggs,
                }
            },
            "size": 0,
            "track_total_hits": False,
        }

    def _build_entity_request(
        self,
        detector: AnomalyDetector,
        entity: Entity,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Build search body for a specific entity's feature aggregation."""
        body = self._build_single_entity_request(detector, start_time, end_time)
        # Add entity filter as additional bool.must term queries
        entity_must = []
        for attr_name, attr_value in entity.attributes.items():
            entity_must.append({"term": {attr_name: attr_value}})
        if entity_must:
            existing = body["query"]["bool"].get("filter", [])
            body["query"]["bool"]["filter"] = existing + entity_must
        return body

    def _build_preview_entities_request(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
        max_entities: int = 1000,
    ) -> Dict[str, Any]:
        """Build composite aggregation query to discover top entities."""
        if not detector.category_fields:
            raise ValueError("Detector has no category_fields")
        sources = [
            {field: {"terms": {"field": field}}} for field in detector.category_fields
        ]
        return {
            "query": self._build_base_query(detector, start_time, end_time),
            "aggs": {
                "top_entities": {
                    "composite": {
                        "sources": sources,
                        "size": max_entities,
                    }
                }
            },
            "size": 0,
            "track_total_hits": False,
        }

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_buckets(
        response: Dict[str, Any], num_features: int
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Extract feature vectors and time ranges from date_histogram buckets.

        Returns (vectors, time_ranges) where vectors is float64[:, num_features].
        Missing sub-aggregation values become NaN.
        """
        agg = response.get("aggregations", {})
        buckets = agg.get("buckets", {}).get("buckets", [])
        if not buckets:
            return np.zeros((0, num_features)), []

        vectors = np.full((len(buckets), num_features), np.nan, dtype=np.float64)
        time_ranges: List[Tuple[int, int]] = []

        for i, b in enumerate(buckets):
            # Bucket key is epoch_millis start; doc_count is number of docs
            start_ms = int(b["key"])
            # end = start + interval. Interval size not explicitly returned,
            # caller should derive from detection_interval if needed.
            time_ranges.append((start_ms, start_ms))  # placeholder; caller can compute end
            for f in range(num_features):
                sub = b.get(f"feature_{f}", {})
                # Handle different aggregation result shapes
                val = _extract_agg_value(sub)
                vectors[i, f] = val

        return vectors, time_ranges

    # ------------------------------------------------------------------ #
    # Imputation
    # ------------------------------------------------------------------ #

    @classmethod
    def impute(
        cls,
        data: np.ndarray,
        option: Optional[ImputationOption] = None,
        feature_names: Optional[List[str]] = None,
        default_fills: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Impute missing values (NaN) in feature matrix.

        Returns (imputed_data, mask) where mask[i, j] is True if the value
        at [i, j] was imputed.
        """
        if data.size == 0:
            return data, None

        mask = np.isnan(data)
        if not np.any(mask):
            return data, mask

        method = ImputationMethod.LINEAR
        if option is not None:
            method = option.method

        imputed = data.copy()

        if method == ImputationMethod.ZERO:
            imputed[mask] = 0.0
        elif method == ImputationMethod.FIXED_VALUES:
            if default_fills is None and option is not None:
                default_fills = option.default_fill or {}
            if feature_names is None or default_fills is None:
                raise ValueError(
                    "FIXED_VALUES imputation requires feature_names and default_fills"
                )
            for j, name in enumerate(feature_names):
                fill = default_fills.get(name, 0.0)
                col_mask = mask[:, j]
                imputed[col_mask, j] = fill
        elif method == ImputationMethod.PREVIOUS:
            for j in range(imputed.shape[1]):
                _forward_fill_1d(imputed[:, j])
        elif method == ImputationMethod.LINEAR:
            for j in range(imputed.shape[1]):
                _linear_interpolate_1d(imputed[:, j])
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        return imputed, mask

    # ------------------------------------------------------------------ #
    # Public async API
    # ------------------------------------------------------------------ #

    async def get_features(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
    ) -> Features:
        """Fetch features for a single-entity detector."""
        body = self._build_single_entity_request(detector, start_time, end_time)
        resp = await self.client.search(index=",".join(detector.indices), body=body)
        vectors, time_ranges = self._parse_buckets(
            resp, detector.base_dimension  # type: ignore[arg-type]
        )
        vectors, mask = self.impute(
            vectors,
            detector.imputation_option,
            detector.enabled_feature_names,
        )
        return Features(
            unprocessed_features=vectors,
            time_ranges=time_ranges,
            imputed_mask=mask,
        )

    async def get_features_for_entity(
        self,
        detector: AnomalyDetector,
        entity: Entity,
        start_time: datetime,
        end_time: datetime,
    ) -> Features:
        """Fetch features for a specific entity in a multi-entity detector."""
        body = self._build_entity_request(detector, entity, start_time, end_time)
        resp = await self.client.search(index=",".join(detector.indices), body=body)
        vectors, time_ranges = self._parse_buckets(
            resp, detector.base_dimension  # type: ignore[arg-type]
        )
        vectors, mask = self.impute(
            vectors,
            detector.imputation_option,
            detector.enabled_feature_names,
        )
        return Features(
            unprocessed_features=vectors,
            time_ranges=time_ranges,
            imputed_mask=mask,
            entity=entity,
        )

    async def get_preview_entities(
        self,
        detector: AnomalyDetector,
        start_time: datetime,
        end_time: datetime,
        max_entities: int = 1000,
    ) -> List[Entity]:
        """Return top entities discovered via composite aggregation."""
        body = self._build_preview_entities_request(
            detector, start_time, end_time, max_entities
        )
        resp = await self.client.search(index=",".join(detector.indices), body=body)
        buckets = (
            resp.get("aggregations", {}).get("top_entities", {}).get("buckets", [])
        )
        entities: List[Entity] = []
        for b in buckets:
            key = b.get("key", {})
            entities.append(Entity(attributes=dict(key)))
        return entities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_agg_value(sub_agg: Dict[str, Any]) -> float:
    """Extract a scalar from an aggregation result bucket."""
    if "value" in sub_agg:
        val = sub_agg["value"]
        if val is None:
            return np.nan
        return float(val)
    if "values" in sub_agg:
        # percentile aggregation returns {"values": {"50.0": 123.4, ...}}
        vals = sub_agg["values"]
        if isinstance(vals, dict):
            # Use the first / median percentile
            keys = sorted(vals.keys())
            v = vals[keys[len(keys) // 2]]
            return float(v) if v is not None else np.nan
        if isinstance(vals, list):
            return float(vals[0]) if vals else np.nan
    # Fallback: some aggregations have doc_count or other structures
    return np.nan


def _forward_fill_1d(arr: np.ndarray) -> None:
    """In-place forward-fill of NaN values in a 1-D array."""
    if arr.size == 0:
        return
    mask = np.isnan(arr)
    if not np.any(mask):
        return
    idx = np.where(~mask, np.arange(arr.size), 0)
    np.maximum.accumulate(idx, out=idx)
    arr[:] = arr[idx]
    # If leading values are still NaN, back-fill them
    if np.isnan(arr[0]):
        # find first non-NaN and fill backwards
        first_valid = np.where(~mask)[0]
        if first_valid.size > 0:
            arr[: first_valid[0]] = arr[first_valid[0]]


def _linear_interpolate_1d(arr: np.ndarray) -> None:
    """In-place linear interpolation of NaN values in a 1-D array."""
    if arr.size == 0:
        return
    mask = np.isnan(arr)
    if not np.any(mask):
        return
    valid = np.where(~mask)[0]
    if valid.size == 0:
        arr[:] = 0.0
        return
    if valid.size == 1:
        arr[:] = arr[valid[0]]
        return
    arr[mask] = np.interp(np.where(mask)[0], valid, arr[valid])
