import os
import joblib
import pandas as pd
import pytest
from destiny_predictor import DestinyPredictor


class DummyModel:
    """Simple mock model returning a fixed prediction."""
    def __init__(self, output):
        self.output = output

    def predict(self, X):
        return [self.output]


class DummyGeneralModel:
    """Mock general model returning a list of predictions."""
    def __init__(self, outputs):
        self.outputs = outputs

    def predict(self, X):
        return [self.outputs]


def test_init_missing_brain(tmp_path, monkeypatch):
    """
    DestinyPredictor should not initialize when the model file is missing.
    is_ready should remain False.
    """
    monkeypatch.chdir(tmp_path)
    dp = DestinyPredictor()

    assert dp.is_ready is False
    assert dp.models == {}
    assert dp.meaning_map == {}


def test_init_success(tmp_path, monkeypatch):
    """
    Test successful loading of a .pkl brain file.
    The predictor should restore both models and the meaning map.
    """
    brain = {
        "meaning_map": {"love": {1: "good"}},
        "Love": DummyModel(1)
    }

    brain_path = tmp_path / "destiny_brain.pkl"
    joblib.dump(brain, brain_path)

    monkeypatch.chdir(tmp_path)
    dp = DestinyPredictor()

    assert dp.is_ready is True
    assert "Love" in dp.models
    assert dp.meaning_map["love"][1] == "good"


@pytest.mark.parametrize("feature_value", [0.5, 1.0])   # <-- pytest now used
def test_predict_without_initialized_model(tmp_path, monkeypatch, feature_value):
    """
    If predictor is not initialized, predict_fortune
    should return an error response instead of crashing.
    """
    monkeypatch.chdir(tmp_path)
    dp = DestinyPredictor()

    out = dp.predict_fortune({"face_lw_ratio": feature_value})

    assert "Error" in out
    assert out["Error"]["sentence"].startswith("AI Models not loaded")


def test_predict_with_general_model(tmp_path, monkeypatch):
    """
    Test prediction using the GENERAL model (multi-output).
    Ensure that each predicted target gets mapped to the correct text.
    """
    class OM:
        ALL_FEATURES = ["x", "y"]

    import destiny_predictor
    monkeypatch.setattr(destiny_predictor, "othermodels", OM)

    brain = {
        "meaning_map": {"career": {3: "excellent"}},
        "GENERAL": {
            "model": DummyGeneralModel([3]),
            "targets": ["Career"]
        }
    }

    brain_path = tmp_path / "destiny_brain.pkl"
    joblib.dump(brain, brain_path)

    monkeypatch.chdir(tmp_path)
    dp = DestinyPredictor()

    out = dp.predict_fortune({"x": 1, "y": 2})

    assert "Career" in out
    assert out["Career"]["sentence"] == "excellent"