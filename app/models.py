from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str
    labels: list[str]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Entity(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class ExtractResponse(BaseModel):
    entities: list[Entity]
