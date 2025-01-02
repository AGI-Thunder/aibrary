from typing import List

from pydantic import BaseModel


class BoundingBox(BaseModel):
    text: str
    left: float
    top: float
    width: float
    height: float


class OCRResponse(BaseModel):
    text: str
    bounding_boxes: List[BoundingBox]
    status: str
