"""Read-only catalogue of real example trials (the ML stage's golden samples).

Each example is a recorded trial from the dataset with its ground-truth label, so a
user can try the model without uploading a file. The model still runs on every request;
nothing here returns a stored prediction.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..schemas import CHANNELS, TrialInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExampleTrial:
    id: str
    source_file: str
    ground_truth_label: str
    trial: TrialInput

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(CHANNELS)
        writer.writerows(self.trial.samples)
        return buffer.getvalue()


class ExampleStore:
    def __init__(self, examples: dict[str, ExampleTrial] | None = None) -> None:
        self._examples = examples or {}

    @classmethod
    def from_directory(cls, directory: Path | None) -> "ExampleStore":
        if directory is None:
            return cls()
        if not directory.is_dir():
            logger.warning("examples_dir_missing")
            return cls()
        examples: dict[str, ExampleTrial] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
                example = ExampleTrial(
                    id=str(raw.get("sample_id") or path.stem),
                    source_file=str(raw.get("source_file", path.name)),
                    ground_truth_label=str(raw["ground_truth_label"]),
                    trial=TrialInput.model_validate(raw["api_request_payload"]),
                )
            except Exception:
                logger.exception("example_trial_invalid file=%s", path.name)
                continue
            examples[example.id] = example
        logger.info("examples_loaded count=%d", len(examples))
        return cls(examples)

    def list(self) -> list[ExampleTrial]:
        return list(self._examples.values())

    def get(self, example_id: str) -> ExampleTrial | None:
        return self._examples.get(example_id)
