"""
Expose the core LLMWrapper and feature modules for direct import.
"""

import os

from aibrary.resources.aibrary_async import AsyncAiBrary
from aibrary.resources.models import Model

__all__ = ["AiBrary", "AsyncAiBrary", "Model"]

base_url = (
    "www.api.aibrary.dev/v0"
    if os.environ.get("DEV_AIBRARY", None)
    else "http://127.0.0.1:8000/v0"
)
