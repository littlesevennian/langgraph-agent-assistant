from enum import Enum
from pydantic import BaseModel,Field, model_validator


class MemoryType(str, Enum):

    BASIC_INFO = "basic_info"

    CAREER = "career"

    SKILL = "skill"

    PROJECT = "project"

    LEARNING_INTEREST = "learning_interest"

    NONE = "none"


class MemoryDecision(BaseModel):

    need_memory_update: bool

    memory_types: list[MemoryType]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    @model_validator(mode="after")
    def validate_memory_decision(self):

        if self.need_memory_update:

            if (
                not self.memory_types
                or MemoryType.NONE in self.memory_types
            ):
                raise ValueError(
                    "need_memory_update=True时，"
                    "memory_types必须包含有效记忆类型"
                )

        else:

            if self.memory_types != [
                MemoryType.NONE
            ]:
                raise ValueError(
                    "need_memory_update=False时，"
                    "memory_types必须为['none']"
                )

        return self