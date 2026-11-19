"""Model lifecycle and availability state, independent of any ML framework."""

from enum import Enum
import json
from pathlib import Path
import threading

from ..contracts import Predictor, PredictorLoader


class ModelStatus(str, Enum):
    NOT_CONFIGURED = "not_configured"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ModelLoadFailure(RuntimeError):
    """Safe lifecycle failure; public API responses do not include its details."""


class ModelUnavailable(RuntimeError):
    def __init__(self, status: ModelStatus):
        self.status = status
        super().__init__(status.value)


class ModelManager:
    def __init__(self) -> None:
        self._status = ModelStatus.NOT_CONFIGURED
        self._predictor: Predictor | None = None
        self._lock = threading.RLock()

    @property
    def status(self) -> ModelStatus:
        with self._lock:
            return self._status

    @property
    def configured(self) -> bool:
        with self._lock:
            return self._status is not ModelStatus.NOT_CONFIGURED

    @property
    def loaded(self) -> bool:
        return self.status is ModelStatus.READY

    def load(self, artifact: Path, loader: PredictorLoader | None) -> None:
        with self._lock:
            self._status = ModelStatus.LOADING
            self._predictor = None
        try:
            if not artifact.exists():
                raise ModelLoadFailure("configured model artifact does not exist")
            if loader is None:
                raise ModelLoadFailure("no ML adapter is registered")
            candidate = loader.load(artifact)
            if not candidate.is_ready():
                raise ModelLoadFailure("predictor adapter is not ready after loading")
            metadata = candidate.get_metadata()
            if not isinstance(metadata, dict):
                raise ModelLoadFailure("predictor metadata must be a dictionary")
            try:
                json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ModelLoadFailure("predictor metadata must be finite JSON data") from exc
            with self._lock:
                self._predictor = candidate
                self._status = ModelStatus.READY
        except Exception as exc:
            with self._lock:
                self._predictor = None
                self._status = ModelStatus.FAILED
            if isinstance(exc, ModelLoadFailure):
                raise
            raise ModelLoadFailure("predictor initialization failed") from exc

    def predictor(self) -> Predictor:
        with self._lock:
            if self._status is not ModelStatus.READY or self._predictor is None:
                raise ModelUnavailable(self._status)
            return self._predictor

    def metadata(self) -> dict[str, object] | None:
        predictor = self.predictor()
        return predictor.get_metadata()

    def clear(self) -> None:
        with self._lock:
            self._predictor = None
            self._status = ModelStatus.NOT_CONFIGURED
            self._status = ModelStatus.NOT_CONFIGURED
