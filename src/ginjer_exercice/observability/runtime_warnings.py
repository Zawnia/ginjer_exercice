from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token

_warnings_var: ContextVar[list[str] | None] = ContextVar("runtime_warnings", default=None)


@contextmanager
def collect_runtime_warnings():
    warnings: list[str] = []
    token: Token = _warnings_var.set(warnings)
    try:
        yield warnings
    finally:
        _warnings_var.reset(token)


def add_runtime_warning(message: str) -> None:
    warnings = _warnings_var.get()
    if warnings is not None:
        warnings.append(message)


def get_runtime_warnings() -> list[str]:
    warnings = _warnings_var.get()
    if warnings is None:
        return []
    return list(warnings)
