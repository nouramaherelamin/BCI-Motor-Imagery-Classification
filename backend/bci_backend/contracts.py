"""Model-facing contracts kept independent of any ML framework."""

from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .schemas import TrialInput


class Prediction(BaseModel):
    """A model result; probabilities are omitted unless the model exposes them."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    probabilities: dict[Literal["Left", "Right"], Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]] | None = None


class Predictor(Protocol):
    """Adapter for a model bundle, including its required preprocessing."""

    def predict(self, trial: TrialInput) -> Prediction:
        """Predict one trial from the validated raw input."""

    def is_ready(self) -> bool:
        """Return whether this adapter can currently serve predictions."""

    def get_metadata(self) -> dict[str, object]:
        """Return safe, serializable model metadata."""


class PredictorLoader(Protocol):
    """Loads a real project-specific predictor from a configured artifact."""

    def load(self, artifact: Path) -> Predictor:
        """Load and initialize a predictor from the given artifact path."""
