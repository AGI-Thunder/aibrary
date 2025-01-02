from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel


class Item(BaseModel):
    label: str
    confidence: float
    x_min: Any
    x_max: Any
    y_min: Any
    y_max: Any


class ObjectDetectionResponse(BaseModel):
    items: List[Item]
