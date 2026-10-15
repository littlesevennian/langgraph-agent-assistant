from typing import Literal
from pydantic import BaseModel,Field


class IntentResult(BaseModel):

    task_type: Literal[
        "resume",
        "project",
        "learning",
        "memory_query",
        "fallback"
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    reason: str