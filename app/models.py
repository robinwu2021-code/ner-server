from typing import Literal

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str

    # labels 可选：空列表 → 服务端自动使用内置双语标签集
    labels: list[str] = Field(
        default_factory=list,
        description=(
            "Entity type labels. Leave empty to use built-in bilingual defaults. "
            "Bilingual pairs (e.g. '人名或姓名' + 'full name of a person') are "
            "automatically expanded to improve recall on Chinese / mixed text."
        ),
    )

    threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence score. "
            "Lower values yield more entities; higher values yield fewer but more precise ones. "
            "Default 0.4 works well for multilingual text."
        ),
    )

    language: Literal["auto", "en", "zh", "ar", "mixed"] = Field(
        default="auto",
        description=(
            "Hint for language-aware processing. "
            "'auto' detects from the text automatically."
        ),
    )


class Entity(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class ExtractResponse(BaseModel):
    entities: list[Entity]
    # Echo back which labels were actually used (useful when labels=[] → defaults applied)
    labels_used: list[str] = Field(default_factory=list)
