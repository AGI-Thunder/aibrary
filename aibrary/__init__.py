"""
Expose the core LLMWrapper and feature modules for direct import.
"""

import os

from aibrary.resources.aibrary_async import AsyncAiBrary
from aibrary.resources.models import Model

__all__ = ["AiBrary", "AsyncAiBrary", "Model"]

if os.environ.get("DEV_AIBRARY", None):
    os.environ["AIBRARY_BASE_URL"] = "http://127.0.0.1:8000/v0"
