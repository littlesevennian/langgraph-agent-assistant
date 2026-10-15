from pydantic import BaseModel, Field


class BasicInfoUpdate(BaseModel):
    name: str | None = None
    age: int | None = None
    location: str | None = None


class CurrentStatusUpdate(BaseModel):
    career: str | None = None


class ProfileUpdate(BaseModel):
    basic_info: BasicInfoUpdate | None = None

    current_status: CurrentStatusUpdate | None = None

    skills: list[str] = Field(
        default_factory=list
    )

    projects: list[str] = Field(
        default_factory=list
    )

    learning_interest: list[str] = Field(
        default_factory=list
    )
