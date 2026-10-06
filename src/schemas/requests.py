"""User requests. Upload IDs are assigned by ingestion, not filesystem paths."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from src.schemas.common import Contract, NonBlank


class Difficulty(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class QuestionType(StrEnum):
    CONCEPTUAL = "conceptual"
    PRACTICAL = "practical"
    SCENARIO = "scenario"
    CODING = "coding"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    SYSTEM_DESIGN = "system_design"
    BEST_PRACTICES = "best_practices"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ENTERPRISE = "enterprise"
    BEHAVIORAL = "behavioral"


class DocumentInput(Contract):
    mode: Literal["document"] = "document"
    document_ids: list[NonBlank] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_documents(self) -> Self:
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("Document IDs must be unique.")
        return self


class WebInput(Contract):
    mode: Literal["web"] = "web"


SourceInput = Annotated[DocumentInput | WebInput, Field(discriminator="mode")]


class InterviewRequest(Contract):
    output: Literal["interview"] = "interview"
    source: SourceInput
    topic: NonBlank
    job_role: NonBlank
    experience_level: Literal["junior", "mid", "senior"]
    difficulty: Difficulty
    question_count: int = Field(gt=0, strict=True)
    question_types: list[QuestionType] = Field(min_length=1)
    domain: NonBlank | None = None

    @model_validator(mode="after")
    def unique_types(self) -> Self:
        if len(set(self.question_types)) != len(self.question_types):
            raise ValueError("Question types must be unique.")
        return self


class PresentationRequest(Contract):
    output: Literal["presentation"] = "presentation"
    source: SourceInput
    topic: NonBlank
    day: int = Field(gt=0, strict=True)
    audience: NonBlank
    slide_count: Literal[10] = 10
    template_id: NonBlank


GenerationRequest = Annotated[
    InterviewRequest | PresentationRequest, Field(discriminator="output")
]
