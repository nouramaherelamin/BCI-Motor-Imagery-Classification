"""Validated, environment-driven application settings (no .env parser required)."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: tuple[str, ...] = ()
    model_artifact: Path | None = None
    examples_dir: Path | None = None
    max_request_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("BCI_ENV must be development, test, or production")
        if not self.host.strip():
            raise ValueError("BCI_HOST cannot be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("BCI_PORT must be between 1 and 65535")
        if self.log_level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("BCI_LOG_LEVEL is not a supported logging level")
        if not 100_000 <= self.max_request_bytes <= 100_000_000:
            raise ValueError("BCI_MAX_REQUEST_BYTES must be between 100000 and 100000000")
        if any(not origin.strip() for origin in self.cors_origins):
            raise ValueError("BCI_CORS_ORIGINS cannot contain empty entries")
        if "*" in self.cors_origins:
            raise ValueError("BCI_CORS_ORIGINS must list explicit origins; wildcard is not allowed")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        raw_origins = env.get("BCI_CORS_ORIGINS", "")
        origins = tuple(item.strip() for item in raw_origins.split(",") if item.strip())
        raw_artifact = env.get("BCI_MODEL_ARTIFACT", "").strip()
        raw_examples = env.get("BCI_EXAMPLES_DIR", "").strip()
        try:
            port = int(env.get("BCI_PORT", "8000"))
            max_request_bytes = int(env.get("BCI_MAX_REQUEST_BYTES", "2000000"))
        except ValueError as exc:
            raise ValueError("BCI_PORT and BCI_MAX_REQUEST_BYTES must be integers") from exc
        return cls(
            environment=env.get("BCI_ENV", "development").strip().lower(),
            host=env.get("BCI_HOST", "127.0.0.1").strip(),
            port=port,
            log_level=env.get("BCI_LOG_LEVEL", "INFO").strip().upper(),
            cors_origins=origins,
            model_artifact=Path(raw_artifact).expanduser() if raw_artifact else None,
            examples_dir=Path(raw_examples).expanduser() if raw_examples else None,
            max_request_bytes=max_request_bytes,
        )
