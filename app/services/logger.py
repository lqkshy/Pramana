"""Centralized logging configuration for the Pramana API."""

import logging

_FORMAT = "%(asctime)s | %(levelname)s | %(filename)s | %(message)s"
_DEFAULT_LEVEL = logging.INFO

_configured = False


def _configure_root() -> None:
    """Idempotently configure the root logger with the standard format."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.basicConfig(level=_DEFAULT_LEVEL, handlers=[handler])

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    _configure_root()
    return logging.getLogger(name)
