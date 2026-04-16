from enum import Enum
from pydantic import BaseModel, Field, field_validator


class AnalysisMode(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD   = "hard"


class AnalyzeRequest(BaseModel):
    argument: str = Field(..., min_length=10, max_length=2000)
    mode:     AnalysisMode = AnalysisMode.MEDIUM
    context:  str | None = Field(None, max_length=1000)
    language: str | None = Field("en", max_length=10)

    @field_validator("argument", "context")
    @classmethod
    def strip_whitespace(cls, v: str | None) -> str | None:
        return v.strip() if v else v
