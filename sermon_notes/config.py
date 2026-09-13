"""Central configuration access for the pipeline.

This module is the ONLY place in the codebase that reads the process
environment (CLAUDE.md §6). Every other module imports accessors from here
rather than touching :data:`os.environ`. Keys are added lazily as the units
that need them arrive; each addition is mirrored in ``.env.example``.
"""

from __future__ import annotations

import os
from typing import overload


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or empty."""


class _Required:
    """Sentinel marking "no default", so ``None`` can itself be a default."""


_REQUIRED = _Required()


@overload
def get(name: str) -> str: ...
@overload
def get(name: str, default: str) -> str: ...
@overload
def get(name: str, default: None) -> str | None: ...


def get(name: str, default: str | None | _Required = _REQUIRED) -> str | None:
    """Return the value of environment variable ``name``.

    With no ``default``, a variable that is unset or empty raises
    :class:`ConfigError` naming the variable. With a ``default`` (including
    ``None``), that default is returned when the variable is unset or empty.
    """
    value = os.environ.get(name)
    if value:
        return value
    if isinstance(default, _Required):
        raise ConfigError(
            f"Required environment variable {name!r} is not set; "
            f"add it to the environment (see .env.example)."
        )
    return default


def get_float(name: str, default: float, *, minimum: float | None = None) -> float:
    """Return environment variable ``name`` parsed as a float.

    The decimal sibling of :func:`get_int`, for knobs denominated in dollars rather
    than counts. Returns ``default`` when the variable is unset or empty. Raises
    :class:`ConfigError` naming the variable and its offending value when the value
    is not a valid number, or when ``minimum`` is given and the parsed value falls
    below it.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name!r} must be a number; got {raw!r}.") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"Environment variable {name!r} must be >= {minimum}; got {value}.")
    return value


def get_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Return environment variable ``name`` parsed as an integer.

    Returns ``default`` when the variable is unset or empty. Raises
    :class:`ConfigError` naming the variable and its offending value when the
    value is not a valid integer, or when ``minimum`` is given and the parsed
    value falls below it.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {name!r} must be an integer; got {raw!r}."
        ) from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"Environment variable {name!r} must be >= {minimum}; got {value}.")
    return value
