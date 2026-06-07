"""Hybrid thresholding model for converting raw RCF scores into anomaly grades."""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from datasketches import kll_floats_sketch
from scipy.special import erf, erfinv

logger = logging.getLogger(__name__)

MIN_SCORE = 0.4
CONFIDENCE = 0.99
DEFAULT_K = 256
DEFAULT_NUM_LOG_NORMAL_QUANTILES = 400
DEFAULT_DOWNSAMPLE_NUM_SAMPLES = 5_000
DEFAULT_DOWNSAMPLE_MAX_NUM_OBSERVATIONS = 50_000
DEFAULT_MIN_PVALUE_THRESHOLD = 0.995
DEFAULT_MAX_RANK_ERROR = 0.0001


@dataclass
class HybridThresholdingModel:
    """
    Python reimplementation of the Java OpenSearch AD
    ``org.opensearch.ad.ml.HybridThresholdingModel``.

    Combines a log-normal distribution model with an empirical CDF
    (approximated by a KLL quantile sketch) for determining anomaly grades.
    """

    min_pvalue_threshold: float = field(default=DEFAULT_MIN_PVALUE_THRESHOLD)
    max_rank_error: float = field(default=DEFAULT_MAX_RANK_ERROR)
    max_score: float = field(default=100.0)
    num_log_normal_quantiles: int = field(default=DEFAULT_NUM_LOG_NORMAL_QUANTILES)
    downsample_num_samples: int = field(default=DEFAULT_DOWNSAMPLE_NUM_SAMPLES)
    downsample_max_num_observations: int = field(
        default=DEFAULT_DOWNSAMPLE_MAX_NUM_OBSERVATIONS
    )
    # Internal sketch parameter.  Java computes K from epsilon;
    # Python datasketches does not expose ``getKFromEpsilon``, so we use a
    # fixed K that yields a reasonable rank error (~0.005).
    _sketch_k: int = field(default=DEFAULT_K, repr=False)

    # Private mutable state (not part of constructor args for simple init)
    _quantile_sketch: kll_floats_sketch = field(init=False, repr=False)
    _num_observations: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.min_pvalue_threshold < 1.0):
            raise ValueError("min_pvalue_threshold must be strictly between 0 and 1")
        if self.max_rank_error <= 0.0:
            raise ValueError("max_rank_error must be positive")
        if self.max_score <= 0.0:
            raise ValueError("max_score must be positive")
        if self.downsample_num_samples <= 1:
            raise ValueError("downsample_num_samples must be > 1")
        if self.downsample_num_samples >= self.downsample_max_num_observations:
            raise ValueError(
                "downsample_num_samples must be < downsample_max_num_observations"
            )
        if self.num_log_normal_quantiles < 0:
            raise ValueError("num_log_normal_quantiles must be non-negative")

        self._quantile_sketch = kll_floats_sketch(self._sketch_k)
        self._num_observations = 0

    # ------------------------------------------------------------------ #
    # Log-normal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_normal_cdf(score: float, mu: float, sigma: float) -> float:
        """
        Log-normal cumulative distribution function.
        Matches Java ``computeLogNormalCdf``.
        """
        return (1.0 + erf((math.log(score) - mu) / (math.sqrt(2.0) * sigma))) / 2.0

    @staticmethod
    def _log_normal_quantile(pvalue: float, mu: float, sigma: float) -> float:
        """
        Log-normal quantile function.
        Matches Java ``computeLogNormalQuantile``.
        """
        return math.exp(mu + math.sqrt(2.0) * sigma * erfinv(2.0 * pvalue - 1.0))

    # ------------------------------------------------------------------ #
    # Training / initialization
    # ------------------------------------------------------------------ #

    def train(self, anomaly_scores: np.ndarray) -> None:
        """
        Initialize the model using a training set of anomaly scores.

        Fits a log-normal distribution to ``log(scores)``, then seeds the
        quantile sketch with ``num_log_normal_quantiles`` samples drawn from
        that distribution up to ``max_score``.
        """
        if anomaly_scores.size == 0:
            return

        logs = np.log(anomaly_scores[anomaly_scores > 0])
        if logs.size == 0:
            return

        mu = float(np.mean(logs))
        sigma = float(np.std(logs, ddof=1)) if logs.size > 1 else 0.0

        max_score_pvalue = self._log_normal_cdf(self.max_score, mu, sigma)
        pvalue_step = max_score_pvalue / (self.num_log_normal_quantiles + 1.0)

        pvalue = pvalue_step
        while pvalue < max_score_pvalue:
            current_score = self._log_normal_quantile(pvalue, mu, sigma)
            self.update(current_score)
            pvalue += pvalue_step

    # ------------------------------------------------------------------ #
    # Online update
    # ------------------------------------------------------------------ #

    def update(self, anomaly_score: float) -> None:
        """Update the empirical CDF with a new anomaly score."""
        self._quantile_sketch.update(float(anomaly_score))
        self._num_observations += 1
        if self._num_observations >= self.downsample_max_num_observations:
            self._downsample()

    # ------------------------------------------------------------------ #
    # Grading
    # ------------------------------------------------------------------ #

    def grade(self, anomaly_score: float) -> float:
        """
        Compute the anomaly grade for a raw RCF score.

        Grade is in ``[0, 1]``; non-zero implies anomalous.
        """
        if anomaly_score <= MIN_SCORE:
            return 0.0

        if self._quantile_sketch.is_empty():
            return 0.0

        scale = 1.0 / (1.0 - self.min_pvalue_threshold)
        pvalue = self._quantile_sketch.get_rank(float(anomaly_score))
        anomaly_grade = scale * (pvalue - self.min_pvalue_threshold)
        if math.isnan(anomaly_grade):
            return 0.0
        return max(0.0, anomaly_grade)

    def confidence(self) -> float:
        """Return model confidence (constant)."""
        return CONFIDENCE

    # ------------------------------------------------------------------ #
    # Downsampling
    # ------------------------------------------------------------------ #

    def _downsample(self) -> None:
        """Replace the sketch with a downsampled version."""
        downsampled = kll_floats_sketch(self._quantile_sketch.k)
        pvalue_step = 1.0 / (self.downsample_num_samples - 1.0)
        pvalue = 0.0
        while pvalue < 1.0:
            score = self._quantile_sketch.get_quantile(pvalue)
            downsampled.update(score)
            pvalue += pvalue_step
        downsampled.update(float(self.max_score))
        self._quantile_sketch = downsampled
        self._num_observations = downsampled.n
        logger.info(
            "Downsampled thresholding model to %d samples (n=%d)",
            self.downsample_num_samples,
            self._num_observations,
        )

    # ------------------------------------------------------------------ #
    # Checkpointing (serialization)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state to a plain dict (JSON-safe)."""
        return {
            "min_pvalue_threshold": self.min_pvalue_threshold,
            "max_rank_error": self.max_rank_error,
            "max_score": self.max_score,
            "num_log_normal_quantiles": self.num_log_normal_quantiles,
            "downsample_num_samples": self.downsample_num_samples,
            "downsample_max_num_observations": self.downsample_max_num_observations,
            "_sketch_k": self._sketch_k,
            "_sketch_b64": base64.b64encode(
                self._quantile_sketch.serialize()
            ).decode(),
            "_num_observations": self._num_observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridThresholdingModel":
        """Restore model state from a dict."""
        # Pop internal fields not handled by the dataclass constructor
        sketch_b64 = data.pop("_sketch_b64")
        num_obs = data.pop("_num_observations", 0)
        # Remove any keys that aren't constructor args (e.g. leftover metadata)
        ctor_fields = {f.name for f in cls.__dataclass_fields__.values() if f.init}
        ctor_kwargs = {k: v for k, v in data.items() if k in ctor_fields}
        instance = cls(**ctor_kwargs)
        instance._quantile_sketch = kll_floats_sketch.deserialize(
            base64.b64decode(sketch_b64)
        )
        instance._num_observations = num_obs
        return instance

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def get_quantiles(self) -> List[float]:
        """Return the list of retained quantile values (for debugging)."""
        # datasketches Python API does not expose an iterator directly;
        # sample at fine granularity to approximate.
        if self._quantile_sketch.is_empty():
            return []
        probs = np.linspace(0.0, 1.0, min(1000, self._quantile_sketch.n))
        return self._quantile_sketch.get_quantiles(probs.tolist())
