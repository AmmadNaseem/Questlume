"""Provider-independent structured interview chains using LCEL."""

from dataclasses import dataclass

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable

from src.prompts.interview import ANSWER_PROMPT, QUESTION_PROMPT, VALIDATION_PROMPT
from src.schemas.interview import QAItem, QAValidationResult
from src.schemas.question_plan import QuestionPlan


@dataclass(frozen=True)
class InterviewChains:
    questions: Runnable
    answers: Runnable
    validation: Runnable


def build_interview_chains(llm: Runnable) -> InterviewChains:
    """JSON parsing works across providers without requiring native tool calling."""
    def chain(prompt, schema):
        parser = PydanticOutputParser(pydantic_object=schema)
        return prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

    return InterviewChains(
        chain(QUESTION_PROMPT, QuestionPlan),
        chain(ANSWER_PROMPT, QAItem),
        chain(VALIDATION_PROMPT, QAValidationResult),
    )
