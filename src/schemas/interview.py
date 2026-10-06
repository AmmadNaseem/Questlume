"""Generated interview answers and evidence integrity validation."""

from typing import Literal, Self

from pydantic import Field, model_validator

from src.schemas.common import Contract, NonBlank
from src.schemas.requests import Difficulty, QuestionType
from src.schemas.sources import SourceReference, validate_references


class QAItem(Contract):
    question: NonBlank
    category: NonBlank
    difficulty: Difficulty
    question_type: QuestionType
    answer: NonBlank
    key_points: list[NonBlank] = Field(min_length=1)
    follow_up_questions: list[NonBlank] = Field(default_factory=list)
    source_ids: list[NonBlank] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class InterviewResult(Contract):
    topic: NonBlank
    source_mode: Literal["document", "web"]
    items: list[QAItem] = Field(min_length=1)
    sources: list[SourceReference] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        validate_references(self.sources, [item.source_ids for item in self.items], self.source_mode)
        questions = [item.question.casefold() for item in self.items]
        if len(set(questions)) != len(questions):
            raise ValueError("Duplicate interview questions are not allowed.")
        return self


class QAValidationResult(Contract):
    approved: bool = Field(strict=True)
    issues: list[NonBlank] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_verdict(self) -> Self:
        if self.approved and self.issues:
            raise ValueError("Approved results must not have unresolved issues.")
        if not self.approved and not self.issues:
            raise ValueError("Rejected results must explain their issues.")
        return self
