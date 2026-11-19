"""Use cases for prediction requests."""

from ..contracts import Prediction
from ..schemas import TrialInput
from .model_manager import ModelManager


class PredictionService:
    def __init__(self, models: ModelManager) -> None:
        self._models = models

    def predict(self, trial: TrialInput) -> Prediction:
        predictor = self._models.predictor()
        return predictor.predict(trial)
