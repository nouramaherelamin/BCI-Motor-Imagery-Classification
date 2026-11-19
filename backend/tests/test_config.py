import pytest

from bci_backend.config import Settings


def test_settings_parse_environment_values():
    settings = Settings.from_env(
        {
            "BCI_ENV": "production",
            "BCI_HOST": "0.0.0.0",
            "BCI_PORT": "9010",
            "BCI_LOG_LEVEL": "debug",
            "BCI_CORS_ORIGINS": "https://app.example, https://admin.example",
            "BCI_MODEL_ARTIFACT": "~/models/mi.bundle",
            "BCI_MAX_REQUEST_BYTES": "3000000",
        }
    )
    assert settings.environment == "production"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9010
    assert settings.log_level == "DEBUG"
    assert settings.cors_origins == ("https://app.example", "https://admin.example")
    assert settings.model_artifact is not None
    assert str(settings.model_artifact).endswith("models/mi.bundle")
    assert settings.max_request_bytes == 3_000_000


@pytest.mark.parametrize(
    "values",
    [
        {"BCI_ENV": "staging"},
        {"BCI_PORT": "70000"},
        {"BCI_PORT": "bad"},
        {"BCI_LOG_LEVEL": "TRACE"},
        {"BCI_MAX_REQUEST_BYTES": "1000"},
        {"BCI_CORS_ORIGINS": "*"},
    ],
)
def test_invalid_environment_settings_fail_fast(values):
    with pytest.raises(ValueError):
        Settings.from_env(values)
