"""
Expose the core LLMWrapper and feature modules for direct import.
"""

import os

from aibrary.resources.aibrary_async import AsyncAiBrary
from aibrary.resources.aibrary_sync import AiBrary
from aibrary.resources.models import Model

__all__ = ["AiBrary", "AsyncAiBrary", "Model"]


def getenv_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("yes", "y", "true", "1", "t")


base_url = (
    "http://127.0.0.1:8000/v0"
    if getenv_bool("DEV_AIBRARY", False)
    else "https://api.aibrary.dev/v0"
)
