"""Evidence-scoped slide generation with bounded content-review revisions."""

import json
import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from pydantic import HttpUrl, TypeAdapter, ValidationError

from src.chains.presentation import PresentationChains
from src.core.config import Settings
from src.pipelines.interview import InsufficientEvidenceError
from src.retrieval.chunking import chunk_to_source
from src.schemas.presentation import PresentationPlan
from src.schemas.requests import DocumentInput, PresentationRequest

logger = logging.getLogger(__name__)


class PresentationQualityError(RuntimeError):
    """No approved plan produced; unapproved slides are not returned."""


class PresentationPipeline:
    def __init__(self, settings: Settings, retriever: Runnable,
                 chains: PresentationChains, *, web_urls: tuple[str, ...] = ()):
        self.settings = settings
        self.retriever = retriever
        self.chains = chains
        self.web_urls = {str(TypeAdapter(HttpUrl).validate_python(url)) for url in web_urls}

    def run(self, request: PresentationRequest) -> PresentationPlan:
        document_mode = isinstance(request.source, DocumentInput)
        mode = "document" if document_mode else "web"
        allowed = set(request.source.document_ids) if document_mode else self.web_urls
        if not allowed:
            raise ValueError("Presentation requires an explicit source selection.")
        sources = {}
        for chunk in self.retriever.invoke(request.topic):
            source = chunk_to_source(chunk)
            identity = source.document_id if source.kind == "document" else str(source.url)
            if source.kind != mode or identity not in allowed:
                raise ValueError("Presentation evidence is outside the selected sources.")
            prospective = {**sources, source.source_id: source}
            encoded = json.dumps([s.model_dump(mode="json") for s in prospective.values()], ensure_ascii=False)
            if len(encoded) <= self.settings.presentation_context_max_chars:
                sources = prospective
        if not sources:
            raise InsufficientEvidenceError("No usable presentation evidence within the context budget.")
        context = json.dumps([s.model_dump(mode="json") for s in sources.values()], ensure_ascii=False)
        limits = dict(max_title_chars=self.settings.presentation_max_title_chars,
                      max_bullets=self.settings.presentation_max_bullets,
                      max_bullet_chars=self.settings.presentation_max_bullet_chars,
                      max_code_chars=self.settings.presentation_max_code_chars,
                      max_notes_chars=self.settings.presentation_max_notes_chars)
        feedback = "First attempt."
        for attempt in range(self.settings.presentation_revision_limit + 1):
            logger.info("presentation_stage stage=draft attempt=%s", attempt)
            try:
                draft = self.chains.draft.invoke(dict(request=request.model_dump_json(), context=context,
                                                     limits=json.dumps(limits), feedback=feedback))
                if not draft.sufficient_evidence:
                    raise InsufficientEvidenceError("Evidence cannot support 10 meaningful slides.")
                cited = {source_id for slide in draft.slides for source_id in slide.source_ids}
                if not cited.issubset(sources):
                    raise ValueError("Slides cite evidence not supplied to generation.")
                plan = PresentationPlan(topic=request.topic, day=request.day, template_id=request.template_id,
                    source_mode=mode, slides=draft.slides, sources=[s for key, s in sources.items() if key in cited])
                titles = [" ".join(s.title.casefold().split()) for s in plan.slides]
                if len(set(titles)) != len(titles):
                    raise ValueError("Slide titles must be distinct.")
                for slide in plan.slides:
                    if len(slide.title) > limits['max_title_chars'] or len(slide.bullets) > limits['max_bullets']:
                        raise ValueError("Slide title or bullet count exceeds content limits.")
                    if any(len(b) > limits['max_bullet_chars'] for b in slide.bullets):
                        raise ValueError("Slide bullets exceed content limits.")
                    if len(slide.code or '') > limits['max_code_chars'] or len(slide.speaker_notes or '') > limits['max_notes_chars']:
                        raise ValueError("Code or speaker notes exceed content limits.")
                logger.info("presentation_stage stage=review attempt=%s", attempt)
                verdict = self.chains.review.invoke(dict(request=request.model_dump_json(),
                    candidate=plan.model_dump_json(), context=context))
                if verdict.approved:
                    logger.info("presentation_stage stage=complete attempt=%s approved=true", attempt)
                    return plan
                feedback = json.dumps(verdict.issues, ensure_ascii=False)
            except InsufficientEvidenceError:
                raise
            except (OutputParserException, ValidationError):
                feedback = "Return valid JSON respecting the schema, 10-slide order and citation requirements."
            except ValueError as error:
                feedback = str(error)
            logger.warning("presentation_revision attempt=%s approved=false", attempt)
        raise PresentationQualityError("Presentation did not pass review within the revision limit.")
