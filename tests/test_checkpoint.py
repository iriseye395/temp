import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.checkpoint_dao import CheckpointDao, CHECKPOINT_INDEX_NAME
from src.models import AnomalyDetector, Feature, IntervalTimeConfiguration, Entity
from src.rcf_wrapper import RCFModel


def make_detector(**kwargs) -> AnomalyDetector:
    defaults = dict(
        name="test-detector",
        time_field="@timestamp",
        indices=["logs-*"],
        features=[Feature(name="cpu", aggregation={"avg": {"field": "cpu"}})],
        detection_interval=IntervalTimeConfiguration(interval=10, unit="minutes"),
        shingle_size=1,
        detector_id="det-123",
    )
    defaults.update(kwargs)
    return AnomalyDetector(**defaults)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def dao(mock_client):
    return CheckpointDao(mock_client)


# ------------------------------------------------------------------ #
# get_checkpoint
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_checkpoint_found(dao, mock_client):
    mock_client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "detectorId": "det-123",
                            "modelId": "det-123",
                            "model_state": {"detector_id": "det-123"},
                        }
                    }
                ]
            }
        }
    )
    result = await dao.get_checkpoint("det-123")
    assert result is not None
    assert result["detectorId"] == "det-123"
    mock_client.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_checkpoint_not_found(dao, mock_client):
    mock_client.search = AsyncMock(return_value={"hits": {"hits": []}})
    result = await dao.get_checkpoint("det-999")
    assert result is None


# ------------------------------------------------------------------ #
# put_checkpoint
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_put_checkpoint(dao, mock_client):
    mock_client.index = AsyncMock(return_value={"_id": "abc", "result": "created"})
    await dao.put_checkpoint(
        model_id="det-123",
        detector_id="det-123",
        model_state={"detector_id": "det-123"},
    )
    mock_client.index.assert_awaited_once()
    call_kwargs = mock_client.index.call_args.kwargs
    assert call_kwargs["index"] == CHECKPOINT_INDEX_NAME
    assert call_kwargs["document"]["modelId"] == "det-123"


# ------------------------------------------------------------------ #
# load_model / save_model round-trip
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_load_model_found(dao, mock_client):
    detector = make_detector()
    model = RCFModel(detector, random_seed=42)
    rng = __import__("numpy").random.RandomState(5)
    for _ in range(200):
        model.process(rng.normal(size=1).tolist())

    checkpoint_state = model.to_dict()
    mock_client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "detectorId": "det-123",
                            "modelId": "det-123",
                            "model_state": checkpoint_state,
                        }
                    }
                ]
            }
        }
    )

    restored = await dao.load_model(detector)
    assert restored is not None
    assert restored.total_updates == model.total_updates


@pytest.mark.asyncio
async def test_load_model_not_found(dao, mock_client):
    mock_client.search = AsyncMock(return_value={"hits": {"hits": []}})
    detector = make_detector()
    restored = await dao.load_model(detector)
    assert restored is None


@pytest.mark.asyncio
async def test_save_model(dao, mock_client):
    mock_client.index = AsyncMock(return_value={"_id": "abc", "result": "created"})
    detector = make_detector()
    model = RCFModel(detector, random_seed=42)
    await dao.save_model(model, detector)
    mock_client.index.assert_awaited_once()
