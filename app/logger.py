
from loguru import logger

from app.config import BASE_DIR

log_dir = BASE_DIR / "logs"

log_dir.mkdir(
    exist_ok=True
)

logger.add(
    str(log_dir / "app.log"),
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="INFO"
)

__all__ = ["logger"]