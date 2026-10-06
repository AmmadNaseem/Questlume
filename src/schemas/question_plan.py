"""Intermediate questions, before generating answers."""

from typing import Self

from pydantic import Field, model_validator

from src.schemas.common import Contract, NonBlank
from src.schemas.requests import Difficulty, QuestionType


class PlannedQuestion(Contract):
    question: NonBlank
    category: NonBlank
    difficulty: Difficulty
    question_type: QuestionType
    source_ids: list[NonBlank] = Field(min_length=1)


class QuestionPlan(Contract):
    sufficient_evidence: bool = Field(strict=True)
    reason: NonBlank
    questions: list[PlannedQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_flag(self) -> Self:
        if self.sufficient_evidence != bool(self.questions):
            raise ValueError("Insufficient evidence must return no questions; sufficient evidence requires questions.")
        return self
