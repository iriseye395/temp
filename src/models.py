"""
Pydantic v2 models mirroring the OpenSearch Anomaly Detection Java classes.

JSON field aliases match the Elasticsearch index mappings used by the
Java plugin so documents can be serialized/deserialized correctly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnomalyDetectorType(str, Enum):
    SINGLE_ENTITY = "SINGLE_ENTITY"
    MULTI_ENTITY = "MULTI_ENTITY"


class ThresholdType(str, Enum):
    # Values inferred from Java plugin usage
    ACTUAL_OVER_EXPECTED = "ACTUAL_OVER_EXPECTED"
    ACTUAL_UNDER_EXPECTED = "ACTUAL_UNDER_EXPECTED"


class ImputationMethod(str, Enum):
    ZERO = "ZERO"
    FIXED_VALUES = "FIXED_VALUES"
    PREVIOUS = "PREVIOUS"
    LINEAR = "LINEAR"


# ---------------------------------------------------------------------------
# Common / Nested models
# ---------------------------------------------------------------------------


class IntervalTimeConfiguration(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    interval: int = Field(..., alias="period.interval", gt=0)
    unit: str = Field(..., alias="period.unit")  # "seconds" or "minutes"


class User(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = Field(default=None, alias="name")
    backend_roles: List[str] = Field(default_factory=list, alias="backend_roles")
    roles: List[str] = Field(default_factory=list, alias="roles")
    custom_attribute_names: List[str] = Field(
        default_factory=list, alias="custom_attribute_names"
    )


class DateRange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    start_time: Optional[datetime] = Field(default=None, alias="start_time")
    end_time: Optional[datetime] = Field(default=None, alias="end_time")


class Entity(BaseModel):
    """
    Multi-entity categorical key.

    In Elasticsearch the `entity` field is stored as a nested array:
        [{"name": "host", "value": "server1"}, ...]

    For ergonomic Python usage we store as a plain dict and provide custom
    (de)serializers.
    """

    model_config = ConfigDict(populate_by_name=True)

    attributes: Dict[str, str] = Field(default_factory=dict)

    def to_es_nested(self) -> List[Dict[str, str]]:
        return [{"name": k, "value": v} for k, v in self.attributes.items()]

    @classmethod
    def from_es_nested(cls, data: List[Dict[str, str]]) -> "Entity":
        return cls(attributes={item["name"]: item["value"] for item in data})


class Feature(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = Field(default=None, alias="feature_id")
    name: str = Field(..., alias="feature_name")
    enabled: bool = Field(default=True, alias="feature_enabled")
    aggregation: Dict[str, Any] = Field(default_factory=dict, alias="aggregation_query")


class ImputationOption(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    method: ImputationMethod = Field(default=ImputationMethod.LINEAR, alias="method")
    default_fill: Optional[Dict[str, float]] = Field(
        default=None, alias="default_fill"
    )


class Condition(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    feature_name: str = Field(..., alias="feature_name")
    threshold_type: str = Field(..., alias="threshold_type")
    operator: str = Field(..., alias="operator")
    value: float = Field(..., alias="value")


class Rule(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    action: Optional[str] = Field(default=None, alias="action")
    conditions: List[Condition] = Field(default_factory=list, alias="conditions")


# ---------------------------------------------------------------------------
# Detector model
# ---------------------------------------------------------------------------


class AnomalyDetector(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detector_id: Optional[str] = Field(default=None, alias="detector_id")
    version: Optional[int] = Field(default=None, alias="version")
    name: str = Field(..., alias="name")
    description: Optional[str] = Field(default=None, alias="description")
    time_field: str = Field(..., alias="time_field")
    indices: List[str] = Field(..., alias="indices")
    features: List[Feature] = Field(..., alias="feature_attributes")
    filter_query: Optional[Dict[str, Any]] = Field(
        default=None, alias="filter_query"
    )
    detection_interval: IntervalTimeConfiguration = Field(
        ..., alias="detection_interval"
    )
    window_delay: Optional[IntervalTimeConfiguration] = Field(
        default=None, alias="window_delay"
    )
    shingle_size: int = Field(default=8, alias="shingle_size", ge=1)
    ui_metadata: Optional[Dict[str, Any]] = Field(
        default=None, alias="ui_metadata"
    )
    schema_version: int = Field(default=0, alias="schema_version")
    last_update_time: Optional[datetime] = Field(
        default=None, alias="last_update_time"
    )
    category_fields: List[str] = Field(default_factory=list, alias="category_field")
    user: Optional[User] = Field(default=None, alias="user")
    result_index: Optional[str] = Field(
        default=".opendistro-anomaly-results", alias="result_index"
    )
    imputation_option: Optional[ImputationOption] = Field(
        default=None, alias="imputation_option"
    )
    recency_emphasis: Optional[int] = Field(
        default=None, alias="recency_emphasis", ge=0, le=100
    )
    season_intervals: Optional[int] = Field(
        default=None, alias="suggested_seasonality", ge=0
    )
    history_intervals: Optional[int] = Field(default=None, alias="history", ge=0)
    custom_result_index_min_size: Optional[int] = Field(
        default=None, alias="result_index_min_size"
    )
    custom_result_index_min_age: Optional[int] = Field(
        default=None, alias="result_index_min_age"
    )
    custom_result_index_ttl: Optional[int] = Field(
        default=None, alias="result_index_ttl"
    )
    flatten_result_index_mapping: Optional[bool] = Field(
        default=None, alias="flatten_result_index_mapping"
    )
    last_ui_breaking_change_time: Optional[datetime] = Field(
        default=None, alias="last_ui_breaking_change_time"
    )
    frequency: Optional[IntervalTimeConfiguration] = Field(
        default=None, alias="forecast_interval"
    )
    detector_type: Optional[AnomalyDetectorType] = Field(
        default=None, alias="detector_type"
    )
    detection_date_range: Optional[DateRange] = Field(
        default=None, alias="detection_date_range"
    )
    rules: List[Rule] = Field(default_factory=list, alias="rules")

    # Derived / transient helpers (not serialized)
    @property
    def is_multi_entity(self) -> bool:
        return self.detector_type == AnomalyDetectorType.MULTI_ENTITY

    @property
    def enabled_features(self) -> List[Feature]:
        return [f for f in self.features if f.enabled]

    @property
    def enabled_feature_names(self) -> List[str]:
        return [f.name for f in self.enabled_features]

    @property
    def base_dimension(self) -> int:
        return len(self.enabled_features)


# ---------------------------------------------------------------------------
# Result / output models
# ---------------------------------------------------------------------------


class FeatureData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    feature_id: Optional[str] = Field(default=None, alias="feature_id")
    feature_name: Optional[str] = Field(default=None, alias="feature_name")
    data: Optional[float] = Field(default=None, alias="data")


class DataByFeatureId(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    feature_id: Optional[str] = Field(default=None, alias="feature_id")
    data: Optional[float] = Field(default=None, alias="data")


class ExpectedValueList(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    likelihood: Optional[float] = Field(default=None, alias="likelihood")
    value_list: Optional[List[DataByFeatureId]] = Field(
        default=None, alias="value_list"
    )


class FeatureImputed(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    feature_id: Optional[str] = Field(default=None, alias="feature_id")
    imputed: bool = Field(default=False, alias="imputed")


class AnomalyResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detector_id: str = Field(..., alias="detector_id")
    task_id: Optional[str] = Field(default=None, alias="task_id")
    anomaly_score: Optional[float] = Field(default=None, alias="anomaly_score")
    anomaly_grade: float = Field(default=0.0, alias="anomaly_grade", ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, alias="confidence", ge=0.0, le=1.0)
    feature_data: List[FeatureData] = Field(default_factory=list, alias="feature_data")
    data_start_time: datetime = Field(..., alias="data_start_time")
    data_end_time: datetime = Field(..., alias="data_end_time")
    execution_start_time: Optional[datetime] = Field(
        default=None, alias="execution_start_time"
    )
    execution_end_time: Optional[datetime] = Field(
        default=None, alias="execution_end_time"
    )
    error: Optional[str] = Field(default=None, alias="error")
    entity: Optional[Entity] = Field(default=None, alias="entity")
    user: Optional[User] = Field(default=None, alias="user")
    schema_version: int = Field(default=0, alias="schema_version")
    model_id: Optional[str] = Field(default=None, alias="model_id")
    approx_anomaly_start_time: Optional[datetime] = Field(
        default=None, alias="approx_anomaly_start_time"
    )
    relevant_attribution: Optional[List[DataByFeatureId]] = Field(
        default=None, alias="relevant_attribution"
    )
    past_values: Optional[List[DataByFeatureId]] = Field(
        default=None, alias="past_values"
    )
    expected_values: Optional[List[ExpectedValueList]] = Field(
        default=None, alias="expected_values"
    )
    threshold: Optional[float] = Field(default=None, alias="threshold")
    feature_imputed: Optional[List[FeatureImputed]] = Field(
        default=None, alias="feature_imputed"
    )

    # Convenience
    @property
    def is_anomaly(self) -> bool:
        return self.anomaly_grade > 0.0


# ---------------------------------------------------------------------------
# Detector internal state
# ---------------------------------------------------------------------------


class DetectorInternalState(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    last_update_time: Optional[datetime] = Field(
        default=None, alias="last_update_time"
    )
    error: Optional[str] = Field(default=None, alias="error")


# ---------------------------------------------------------------------------
# Thresholding intermediate result (not persisted to ES directly)
# ---------------------------------------------------------------------------


class ThresholdingResult(BaseModel):
    """Intermediate result produced by the HybridThresholdingModel."""

    model_config = ConfigDict(populate_by_name=True)

    grade: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rcf_score: float = Field(default=0.0)
    total_updates: int = Field(default=0)
    relative_index: Optional[int] = Field(default=None)
    relevant_attribution: Optional[List[float]] = Field(default=None)
    past_values: Optional[List[float]] = Field(default=None)
    expected_values_list: Optional[List[float]] = Field(default=None)
    likelihood_of_values: Optional[float] = Field(default=None)
    threshold: Optional[float] = Field(default=None)
    number_of_trees: Optional[int] = Field(default=None)
    actual: Optional[List[float]] = Field(default=None)
    is_feature_imputed: Optional[List[bool]] = Field(default=None)
