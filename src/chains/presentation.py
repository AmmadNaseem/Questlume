"""LCEL presentation chains reuse the provider-independent fallback Runnable."""

from dataclasses import dataclass

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from src.prompts.presentation import DRAFT_PROMPT, REVIEW_PROMPT
from src.schemas.interview import QAValidationResult
from src.schemas.presentation import PresentationDraft


@dataclass(frozen=True)
class PresentationChains:
    draft: Runnable
    review: Runnable


def build_presentation_chains(llm: Runnable, *, parsed_draft: Runnable | None = None,
                              parsed_review: Runnable | None = None) -> PresentationChains:
    draft_parser = PydanticOutputParser(pydantic_object=PresentationDraft)
    review_parser = PydanticOutputParser(pydantic_object=QAValidationResult)
    return PresentationChains(
        DRAFT_PROMPT.partial(format_instructions=draft_parser.get_format_instructions()) | (
            parsed_draft if parsed_draft is not None else llm | draft_parser),
        REVIEW_PROMPT.partial(format_instructions=review_parser.get_format_instructions()) | (
            parsed_review if parsed_review is not None else llm | review_parser),
    )
