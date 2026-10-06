"""Evidence-scoped slide generation with bounded content-review revisions."""

import json
import logging
from collections.abc import Callable

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

    def __init__(self, message: str, *, reason: str = 'review'):
        super().__init__(message)
        self.reason = reason


class PresentationPipeline:
    def __init__(self, settings: Settings, retriever: Runnable,
                 chains: PresentationChains, *, web_urls: tuple[str, ...] = ()):
        self.settings = settings
        self.retriever = retriever
        self.chains = chains
        self.web_urls = {str(TypeAdapter(HttpUrl).validate_python(url)) for url in web_urls}

    def _review(self, request, plan, context, progress):
        feedback = 'Return the required verdict JSON.'
        for attempt in range(self.settings.presentation_format_retry_limit + 1):
            try:
                return self.chains.review.invoke(dict(request=request.model_dump_json(),
                    candidate=plan.model_dump_json(), context=context, format_feedback=feedback))
            except (OutputParserException, ValidationError):
                logger.warning('presentation_review_format attempt=%s valid=false', attempt)
                if attempt >= self.settings.presentation_format_retry_limit:
                    raise PresentationQualityError('Reviewer output could not be validated.', reason='format') from None
                feedback = ('The last verdict did not match the schema. Return ONLY the complete JSON object '
                            'with approved: boolean and issues: array of strings. No slide draft or Markdown. '
                            'If approved is true, issues must be empty; otherwise explain the corrections.')
                progress('Reviewer returned invalid JSON; retrying the review while keeping the validated slides')

    def run(self, request: PresentationRequest, *, progress: Callable[[str], None] = lambda message: None) -> PresentationPlan:
        progress('Retrieving evidence for the presentation')
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
        previous_candidate = 'No previous draft.'
        review_issues = []
        failure_kind = 'review'
        for attempt in range(self.settings.presentation_revision_limit + 1):
            progress(f'Drafting 10 slides (attempt {attempt + 1})')
            logger.info("presentation_stage stage=draft attempt=%s", attempt)
            try:
                draft = self.chains.draft.invoke(dict(request=request.model_dump_json(), context=context,
                                                     limits=json.dumps(limits), feedback=feedback,
                                                     previous_candidate=previous_candidate))
                previous_candidate = draft.model_dump_json()
                if not draft.sufficient_evidence:
                    raise InsufficientEvidenceError("Evidence cannot support 10 meaningful slides.")
                progress('Checking slide count, content limits and citations')
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
                progress('Reviewing all 10 slides against their sources')
                verdict = self._review(request, plan, context, progress)
                if verdict.approved:
                    logger.info("presentation_stage stage=complete attempt=%s approved=true", attempt)
                    progress('Presentation plan completed and approved')
                    return plan
                review_issues = verdict.issues
                failure_kind = 'review'
                feedback = json.dumps({'review_issues': review_issues}, ensure_ascii=False)
                progress('The reviewer found unsupported or unsuitable content; applying corrections')
            except InsufficientEvidenceError:
                raise
            except (OutputParserException, ValidationError) as error:
                failure_kind = 'format'
                validation = error if isinstance(error, ValidationError) else error.__cause__
                schema_errors = []
                if isinstance(validation, ValidationError):
                    allowed_fields = {'slides', 'number', 'layout', 'purpose', 'title', 'bullets',
                                      'code', 'speaker_notes', 'source_ids', 'sufficient_evidence',
                                      'reason', 'approved', 'issues'}
                    for issue in validation.errors(include_input=False):
                        location = [part for part in issue['loc'] if isinstance(part, int) or part in allowed_fields]
                        schema_errors.append({'field': location, 'error_type': issue['type']})
                feedback = json.dumps({'review_issues': review_issues,
                    'correction': 'Return the COMPLETE PresentationDraft JSON object, not a patch, explanation or just the slides array. Respect the exact schema, 10-slide order and cited sources.',
                    'schema_errors': schema_errors})
                progress('The model returned invalid structured output; requesting a complete corrected JSON draft')
            except ValueError as error:
                failure_kind = 'constraints'
                feedback = json.dumps({'review_issues': review_issues, 'constraint_correction': str(error)})
                progress('The draft failed slide structure, citation or content-limit checks; requesting corrections')
            logger.warning("presentation_revision attempt=%s approved=false failure_kind=%s", attempt, failure_kind)
            if attempt < self.settings.presentation_revision_limit:
                progress('Quality checks require another attempt; revising the slides')
        raise PresentationQualityError("Presentation did not pass review within the revision limit.", reason=failure_kind)
