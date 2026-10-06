"""PDF interview orchestration with bounded revision and evidence checks."""

import json
import logging
from collections.abc import Callable

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from src.chains.interview import InterviewChains
from src.core.config import Settings
from src.retrieval.chunking import chunk_to_source
from src.schemas.interview import InterviewResult
from src.schemas.requests import DocumentInput, InterviewRequest, WebInput
from src.schemas.sources import DocumentSource, WebSource

logger = logging.getLogger(__name__)


class InsufficientEvidenceError(ValueError):
    """Retrieved evidence cannot support the requested output."""


class InterviewQualityError(RuntimeError):
    """No approved interview was produced within the revision budget."""


def _context(sources: list[DocumentSource | WebSource]) -> str:
    return json.dumps([source.model_dump(mode="json") for source in sources], ensure_ascii=False)


class InterviewPipeline:
    """Shared generation workflow; source-specific adapters enforce evidence scope."""
    def __init__(self, settings: Settings, retriever: Runnable, chains: InterviewChains):
        self.settings = settings
        self.retriever = retriever
        self.chains = chains

    def _retrieve(self, query: str, allowed: set[str]) -> list[DocumentSource | WebSource]:
        sources: list[DocumentSource | WebSource] = []
        seen: set[str] = set()
        for chunk in self.retriever.invoke(query):
            source = chunk_to_source(chunk)
            identity = source.document_id if source.kind == "document" else str(source.url)
            if source.kind != self.source_mode or identity not in allowed:
                raise ValueError("Retriever returned evidence outside the selected sources.")
            if source.source_id in seen:
                continue
            if len(_context([*sources, source])) > self.settings.interview_context_max_chars:
                continue
            seen.add(source.source_id)
            sources.append(source)
        if not sources:
            raise InsufficientEvidenceError("No usable evidence within the context budget.")
        return sources

    def run(self, request: InterviewRequest, *, progress: Callable[[str], None] = lambda message: None) -> InterviewResult:
        allowed = self._allowed_sources(request)
        if request.question_count > self.settings.interview_max_questions:
            raise ValueError("Requested question count exceeds the configured limit.")
        progress('Retrieving evidence for the interview')
        initial = self._retrieve(request.topic, allowed)
        payload = request.model_dump_json()
        feedback = "First attempt."
        for attempt in range(self.settings.interview_revision_limit + 1):
            progress(f'Planning {request.question_count} questions (attempt {attempt + 1})')
            logger.info("interview_stage stage=questions attempt=%s", attempt)
            try:
                plan = self.chains.questions.invoke(dict(request=payload, context=_context(initial), feedback=feedback))
                if not plan.sufficient_evidence:
                    raise InsufficientEvidenceError("Evidence cannot support the requested interview.")
                if len(plan.questions) != request.question_count:
                    raise ValueError("Question count does not match the request.")
                initial_ids = {source.source_id for source in initial}
                items = []
                all_sources: dict[str, DocumentSource | WebSource] = {}
                normalized_questions: set[str] = set()
                for number, question in enumerate(plan.questions, 1):
                    normalized = " ".join(question.question.casefold().split())
                    if normalized in normalized_questions:
                        raise ValueError("Duplicate planned questions.")
                    normalized_questions.add(normalized)
                    if question.difficulty != request.difficulty or question.question_type not in request.question_types:
                        raise ValueError("Question parameters do not match the request.")
                    if not set(question.source_ids).issubset(initial_ids):
                        raise ValueError("Question cites evidence not supplied to planning.")
                    # Retain planning evidence, then add per-question retrieval where budget permits.
                    progress(f'Retrieving evidence for question {number} of {request.question_count}')
                    evidence = [s for s in initial if s.source_id in question.source_ids]
                    for source in self._retrieve(question.question, allowed):
                        if source.source_id not in {s.source_id for s in evidence} and len(_context([*evidence, source])) <= self.settings.interview_context_max_chars:
                            evidence.append(source)
                    logger.info("interview_stage stage=answer attempt=%s", attempt)
                    progress(f'Writing answer {number} of {request.question_count}')
                    answer = self.chains.answers.invoke(dict(request=payload, question=question.model_dump_json(), context=_context(evidence), feedback=feedback))
                    for field in ("question", "category", "difficulty", "question_type"):
                        if getattr(answer, field) != getattr(question, field):
                            raise ValueError("Answer changed planned question fields.")
                    if not set(answer.source_ids).issubset({s.source_id for s in evidence}):
                        raise ValueError("Answer cites unavailable evidence.")
                    progress(f'Reviewing answer {number} of {request.question_count}')
                    verdict = self.chains.validation.invoke(dict(request=payload, candidate=answer.model_dump_json(), context=_context(evidence)))
                    if not verdict.approved:
                        feedback = json.dumps(verdict.issues, ensure_ascii=False)
                        raise InterviewQualityError("Answer failed grounding review.")
                    items.append(answer)
                    all_sources.update({s.source_id: s for s in evidence if s.source_id in answer.source_ids})
                result = InterviewResult(topic=request.topic, source_mode=self.source_mode, items=items, sources=list(all_sources.values()))
                logger.info("interview_stage stage=complete attempt=%s approved=true", attempt)
                progress('Interview completed and approved')
                return result
            except InsufficientEvidenceError:
                raise
            except (OutputParserException, ValidationError):
                feedback = "Return valid JSON matching the supplied schema and constraints."
            except ValueError as error:
                feedback = str(error)
            except InterviewQualityError:
                pass
            logger.warning("interview_revision attempt=%s approved=false", attempt)
            if attempt < self.settings.interview_revision_limit:
                progress('Quality checks require another attempt; revising the interview')
        raise InterviewQualityError("Interview failed validation after the configured revision limit.")

    def _allowed_sources(self, request: InterviewRequest) -> set[str]:
        raise NotImplementedError


class PDFInterviewPipeline(InterviewPipeline):
    source_mode = "document"

    def _allowed_sources(self, request: InterviewRequest) -> set[str]:
        if not isinstance(request.source, DocumentInput):
            raise ValueError("PDF pipeline requires document mode.")
        return set(request.source.document_ids)


class WebInterviewPipeline(InterviewPipeline):
    source_mode = "web"

    def __init__(self, settings: Settings, retriever: Runnable, chains: InterviewChains, urls: list[str]):
        super().__init__(settings, retriever, chains)
        from pydantic import HttpUrl, TypeAdapter
        self.urls = {str(TypeAdapter(HttpUrl).validate_python(url)) for url in urls}
        if not self.urls:
            raise ValueError("Web pipeline requires fetched source URLs.")

    def _allowed_sources(self, request: InterviewRequest) -> set[str]:
        if not isinstance(request.source, WebInput):
            raise ValueError("Web pipeline requires web mode.")
        return self.urls
