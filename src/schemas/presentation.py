"""Slide content contracts, separate from PowerPoint rendering."""

from typing import Literal, Self

from pydantic import Field, model_validator

from src.schemas.common import Contract, NonBlank
from src.schemas.sources import SourceReference, validate_references


class SlideSpec(Contract):
    number: int = Field(ge=1, le=10, strict=True)
    layout: Literal["cover", "concept", "comparison", "code", "application", "challenge", "takeaways"]
    purpose: NonBlank
    title: NonBlank
    bullets: list[NonBlank] = Field(default_factory=list)
    code: NonBlank | None = None
    speaker_notes: NonBlank | None = None
    source_ids: list[NonBlank] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if self.layout != "cover" and not (self.bullets or self.code):
            raise ValueError("Content slides require bullets or code.")
        if self.layout != "cover" and not self.source_ids:
            raise ValueError("Content slides require evidence references.")
        if self.layout == "code" and self.code is None:
            raise ValueError("Code layouts require a code example.")
        return self


class PresentationPlan(Contract):
    topic: NonBlank
    day: int | None = Field(default=None, gt=0, strict=True)
    source_mode: Literal["document", "web"]
    template_id: NonBlank
    slides: list[SlideSpec] = Field(min_length=10, max_length=10)
    sources: list[SourceReference] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if [slide.number for slide in self.slides] != list(range(1, 11)):
            raise ValueError("Slides must be ordered and numbered 1 through 10.")
        if self.slides[0].layout != "cover" or any(s.layout == "cover" for s in self.slides[1:]):
            raise ValueError("Only the first slide must use the cover layout.")
        validate_references(self.sources, [slide.source_ids for slide in self.slides], self.source_mode)
        return self


class PresentationDraft(Contract):
    """LLM supplies slide content; application supplies trusted source records."""

    sufficient_evidence: bool = Field(strict=True)
    reason: NonBlank
    slides: list[SlideSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.sufficient_evidence and len(self.slides) != 10:
            raise ValueError("Supported presentation drafts require exactly 10 slides.")
        if not self.sufficient_evidence and self.slides:
            raise ValueError("Insufficient evidence must produce no slides.")
        return self
