"""The project logger, configured once at import.

Import :func:`get_logger` (or the module-level :data:`logger`) instead of
using ``print`` or the ``logging`` root (CLAUDE.md §6). The log level is read
through :mod:`sermon_notes.config`, never from the environment directly.
"""

from __future__ import annotations

import logging

from sermon_notes import config

_LOGGER_NAME = "sermon_notes"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _configure() -> logging.Logger:
    """Configure and return the project logger, idempotently."""
    log = logging.getLogger(_LOGGER_NAME)
    level = config.get("LOG_LEVEL", "info").upper()
    log.setLevel(getattr(logging, level, logging.INFO))
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        log.addHandler(handler)
    log.propagate = False
    return log


logger = _configure()


def get_logger() -> logging.Logger:
    """Return the configured project logger."""
    return logger
