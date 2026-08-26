"""
Structured JSON logging configuration for Pramana.

Configures the standard library logging module with a JSON formatter
so that logs are machine-readable in production deployments.
"""
import logging

__all__ = ["get_logger"]

def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured for Pramana."""
    # TODO: attach JSON formatter
    return logging.getLogger(name)


def configure_logging() -> None:
    """Configure root logging for the application."""
    # TODO: implement
    pass
