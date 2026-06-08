"""Random Cut Forest wrapper using ``krcf`` with hybrid thresholding."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from krcf import RandomCutForest, RandomCutForestOptions

from src.models import AnomalyDetector, Entity
from src.thresholding import HybridThresholdingModel

logger = logging.getLogger(__name__)

# Java defaults (from TimeSeriesSettings / AnomalyDetectorSettings)
DEFAULT_NUM_TREES = 50
DEFAULT_SAMPLE_SIZE = 256
DEFAULT_MIN_SAMPLES = 32
DEFAULT_TIME_DECAY = 0.0001
DEFAULT_INITIAL_ACCEPT_FRACTION = 1.0


@dataclass
class AnomalyDescriptor:
    """
    Analogous to Java ``com.amazon.randomcutforest.parkservices.AnomalyDescriptor``.

    Holds the result of processing a single data point through the
    ThresholdedRandomCutForest.
    """

    rcf_score: float = 0.0
    anomaly_grade: float = 0.0
    confidence: float = 0.0
    total_updates: int = 0
    relative_index: int = 0
    relevant_attribution: Optional[List[float]] = None
    past_values: Optional[List[float]] = None
    expected_values_list: Optional[List[float]] = None
    likelihood_of_values: Optional[float] = None
    threshold: Optional[float] = None
    number_of_trees: int = 0
    is_output_ready: bool = False
    is_imputed: bool = False
    is_anomaly: bool = False
    actual: Optional[List[float]] = None
    is_feature_imputed: Optional[List[bool]] = None


class RCFModel:
    """
    Python equivalent of Java's ``ThresholdedRandomCutForest`` + ``ADModelManager``.

    Wraps a ``krcf.RandomCutForest`` for scoring and a
    ``HybridThresholdingModel`` for thresholding / grading.
    """

    def __init__(
        self,
        detector: AnomalyDetector,
        *,
        entity: Optional[Entity] = None,
        num_trees: int = DEFAULT_NUM_TREES,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        time_decay: float = DEFAULT_TIME_DECAY,
        initial_accept_fraction: float = DEFAULT_INITIAL_ACCEPT_FRACTION,
        random_seed: int = 0,
        anomaly_rate: Optional[float] = None,
    ):
        self.detector = detector
        self.entity = entity
        self._num_trees = num_trees
        self._sample_size = sample_size
        self._min_samples = min_samples

        base_dimension = detector.base_dimension
        shingle_size = detector.shingle_size

        if anomaly_rate is None:
            # Java default: anomalyRate(1 - thresholdMinPvalue)
            # We compute dynamically from the default thresholding model
            anomaly_rate = 1.0 - 0.995

        opts: Dict[str, Any] = {
            "dimensions": base_dimension,
            "shingle_size": shingle_size,
            "num_trees": num_trees,
            "sample_size": sample_size,
            "time_decay": time_decay,
            "output_after": min_samples,
            "initial_accept_fraction": initial_accept_fraction,
            "random_seed": random_seed,
            "anomaly_rate": anomaly_rate,
            "compact": True,
        }
        self._forest = RandomCutForest(RandomCutForestOptions(**opts))
        self._thresholding_model = HybridThresholdingModel()

        logger.info(
            "RCFModel created: dim=%d, shingle=%d, trees=%d, sample=%d, entity=%s",
            base_dimension,
            shingle_size,
            num_trees,
            sample_size,
            entity.attributes if entity else "single",
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def dimensions(self) -> int:
        return self._forest.dimensions()

    @property
    def shingle_size(self) -> int:
        return self._forest.shingle_size()

    @property
    def total_updates(self) -> int:
        return self._forest.entries_seen()

    def is_output_ready(self) -> bool:
        return self._forest.is_output_ready()

    # ------------------------------------------------------------------ #
    # Training / cold start
    # ------------------------------------------------------------------ #

    def train(self, data_points: np.ndarray) -> None:
        """
        Cold-start training: feed historical data points and fit the
        thresholding model to the resulting RCF scores.
        """
        if data_points.ndim != 2:
            raise ValueError("data_points must be 2-D array (samples, features)")

        scores: List[float] = []
        for point in data_points:
            self._forest.update(point.tolist())
            if self._forest.is_output_ready():
                scores.append(self._forest.score(point.tolist()))

        if scores:
            self._thresholding_model.train(np.array(scores, dtype=np.float64))

    # ------------------------------------------------------------------ #
    # Real-time / streaming process
    # ------------------------------------------------------------------ #

    def process(self, point: List[float], timestamp_secs: int = 0) -> AnomalyDescriptor:
        """
        Process a single data point through the forest and thresholding model.

        Parameters
        ----------
        point: 1-D vector of length ``detector.base_dimension``.
        timestamp_secs: Unix epoch seconds (for parity with Java; not used).

        Returns
        -------
        AnomalyDescriptor with score, grade, attribution, etc.
        """
        if not self._forest.is_output_ready():
            # Accumulate without scoring until minimum samples reached
            self._forest.update(point)
            return AnomalyDescriptor(
                total_updates=self._forest.entries_seen(),
                is_output_ready=False,
            )

        # Score BEFORE updating so the score reflects how anomalous the point
        # is w.r.t. the current (not yet updated) model.
        rcf_score = self._forest.score(point)
        anomaly_grade = self._thresholding_model.grade(rcf_score)
        confidence = self._thresholding_model.confidence()
        attribution = self._forest.attribution(point)
        total_updates = self._forest.entries_seen()

        # Update the forest and thresholding model with the observed point
        self._forest.update(point)
        self._thresholding_model.update(rcf_score)

        return AnomalyDescriptor(
            rcf_score=rcf_score,
            anomaly_grade=anomaly_grade,
            confidence=confidence,
            total_updates=total_updates,
            relative_index=0,
            relevant_attribution=_flatten_attribution(attribution, self.dimensions),
            number_of_trees=self._num_trees,
            is_output_ready=True,
            is_anomaly=anomaly_grade > 0.0,
        )

    def process_batch(
        self, data_points: np.ndarray, start_timestamp_secs: int = 0
    ) -> List[AnomalyDescriptor]:
        """Process a batch of points sequentially."""
        results: List[AnomalyDescriptor] = []
        for i, point in enumerate(data_points):
            desc = self.process(point.tolist(), start_timestamp_secs + i)
            results.append(desc)
        return results

    # ------------------------------------------------------------------ #
    # Preview scoring (no model mutation)
    # ------------------------------------------------------------------ #

    def get_preview_results(
        self, data_points: np.ndarray
    ) -> List[AnomalyDescriptor]:
        """
        Score a batch without updating the underlying model.

        Creates a temporary clone of the forest, scores all points, and
        applies the current thresholding model.
        """
        temp_forest = RandomCutForest.from_json(self._forest.to_json())
        results: List[AnomalyDescriptor] = []
        for point in data_points:
            if not temp_forest.is_output_ready():
                temp_forest.update(point.tolist())
                results.append(
                    AnomalyDescriptor(
                        total_updates=temp_forest.entries_seen(),
                        is_output_ready=False,
                    )
                )
                continue
            rcf_score = temp_forest.score(point.tolist())
            anomaly_grade = self._thresholding_model.grade(rcf_score)
            confidence = self._thresholding_model.confidence()
            attribution = temp_forest.attribution(point.tolist())
            temp_forest.update(point.tolist())
            results.append(
                AnomalyDescriptor(
                    rcf_score=rcf_score,
                    anomaly_grade=anomaly_grade,
                    confidence=confidence,
                    total_updates=temp_forest.entries_seen(),
                    relevant_attribution=_flatten_attribution(
                        attribution, temp_forest.dimensions()
                    ),
                    number_of_trees=self._num_trees,
                    is_output_ready=True,
                    is_anomaly=anomaly_grade > 0.0,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state for checkpoint persistence."""
        return {
            "detector_id": self.detector.detector_id,
            "entity": self.entity.to_es_nested() if self.entity else None,
            "forest_msgpack_b64": base64.b64encode(
                self._forest.to_msgpack()
            ).decode(),
            "thresholding_model": self._thresholding_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], detector: AnomalyDetector) -> "RCFModel":
        """Restore model state from a checkpoint dict."""
        entity_data = data.get("entity")
        entity = Entity.from_es_nested(entity_data) if entity_data else None

        instance = cls(detector, entity=entity)
        forest_bytes = base64.b64decode(data["forest_msgpack_b64"])
        instance._forest = RandomCutForest.from_msgpack(forest_bytes)
        instance._thresholding_model = HybridThresholdingModel.from_dict(
            data["thresholding_model"]
        )
        return instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_attribution(
    attribution: Dict[str, List[float]], dimensions: int
) -> List[float]:
    """
    Convert ``krcf`` attribution ``{"high": [...], "low": [...]}`` to a
    single vector of attribution magnitudes.

    The Java plugin uses ``normalizeAttribution(rcf, descriptor.getRelevantAttribution())``.
    Since krcf does not expose the exact same attribution vector, we approximate
    by summing high + low per dimension.
    """
    high = attribution.get("high", [])
    low = attribution.get("low", [])
    # Ensure lengths match dimensions (pad with zeros if needed)
    high = high + [0.0] * (dimensions - len(high))
    low = low + [0.0] * (dimensions - len(low))
    return [high[i] + low[i] for i in range(min(dimensions, len(high), len(low)))]
