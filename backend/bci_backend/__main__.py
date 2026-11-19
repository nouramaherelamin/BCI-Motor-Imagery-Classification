"""Run the development server using validated environment configuration."""

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "bci_backend.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.environment == "development",
    )


if __name__ == "__main__":
    main()
