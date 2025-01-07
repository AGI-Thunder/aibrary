from pydantic import BaseModel


class TranslationsResponse(BaseModel):
    text: str
    status: str