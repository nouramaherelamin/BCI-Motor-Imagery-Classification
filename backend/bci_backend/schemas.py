"""HTTP schemas based only on currently documented input evidence."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

CHANNELS = ("FZ", "C3", "CZ", "C4")
SAMPLES_PER_FILE = 2500
SAMPLE_RATE_HZ = 250.0
Sample = tuple[
    Annotated[float, Field(allow_inf_nan=False)],
    Annotated[float, Field(allow_inf_nan=False)],
    Annotated[float, Field(allow_inf_nan=False)],
    Annotated[float, Field(allow_inf_nan=False)],
]


class TrialInput(BaseModel):
    """One repository-shaped 10-second EEG file, in its recorded channel order."""

    model_config = ConfigDict(extra="forbid")

    sampling_rate_hz: Literal[250.0] = Field(description="Provisional nominal sampling rate in Hz.")
    channels: tuple[Literal["FZ"], Literal["C3"], Literal["CZ"], Literal["C4"]] = Field(
        description="Channel names in sample-column order; the current provisional order is required."
    )
    samples: Annotated[
        list[Sample],
        Field(
            min_length=SAMPLES_PER_FILE,
            max_length=SAMPLES_PER_FILE,
            description="Exactly 2500 rows by four finite EEG values in channel order.",
        ),
    ]


class PredictionResponse(BaseModel):
    predicted_label: Literal["Left", "Right"]
    model_version: str
    probabilities: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_status: Literal["not_configured", "loading", "ready", "failed"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    configured: bool
    loaded: bool
    status: Literal["not_configured", "loading", "ready", "failed"]
    model_name: str | None
    model_version: str | None
    capabilities: list[str] | None
    expected_input: dict[str, object]
    metadata: dict[str, object] | None


class ApiError(BaseModel):
    code: str
    message: str
    details: list[dict[str, object]] | None = None


class ErrorResponse(BaseModel):
    error: ApiError


class ExampleSummary(BaseModel):
    id: str
    source_file: str
    ground_truth_label: str


class ExampleDetail(ExampleSummary):
    payload: TrialInput
