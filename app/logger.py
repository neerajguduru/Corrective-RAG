from pathlib import Path

from loguru import logger

log_dir = Path("logs")

log_dir.mkdir(
    exist_ok=True
)

logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="INFO"
)

__all__ = ["logger"]