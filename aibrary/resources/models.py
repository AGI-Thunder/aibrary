from typing import List, Optional

from pydantic import BaseModel


class AiModelsPricingItem(BaseModel):
    unit_type: str
    created_at: str
    update_at: str
    price_per_input_unit: float
    price_per_output_unit: float


class Model(BaseModel):
    provider: str
    category: str
    model_name: str
    ai_models_pricing: List[AiModelsPricingItem]
    status: str
    quality: Optional[str] = None
    size: Optional[str] = None
