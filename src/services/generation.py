"""Request-scoped orchestration without UI dependencies or shared evidence caches."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile

from src.chains.interview import build_interview_chains
from src.chains.presentation import build_presentation_chains
from src.core.config import PROJECT_ROOT, Settings
from src.core.llm_factory import get_llm, ContextLimitError
from src.export.pptx_renderer import render_pptx
from src.ingestion.pdf_loader import load_pdfs, PDFIngestionError
from src.ingestion.web_search import get_search_tool
from src.pipelines.interview import PDFInterviewPipeline, WebInterviewPipeline, InsufficientEvidenceError
from src.pipelines.presentation import PresentationPipeline
from src.pipelines.web_research import research_web
from src.retrieval.chunking import split_pages
from src.retrieval.embeddings import get_embeddings
from src.retrieval.retriever import build_retriever
from src.retrieval.vector_store import build_document_index, build_web_index
from src.schemas.interview import InterviewResult
from src.schemas.presentation import PresentationPlan
from src.schemas.requests import DocumentInput, InterviewRequest, PresentationRequest


@dataclass(frozen=True)
class Upload:
    name: str
    content: bytes


@dataclass(frozen=True)
class GenerationResult:
    output: InterviewResult | PresentationPlan
    warnings: tuple[str, ...]


@contextmanager
def staged_pdfs(uploads: Sequence[Upload], settings: Settings) -> Iterator[list[Path]]:
    """Validate a batch before disk writes, isolate it and clean up on any exit."""
    if not uploads or len(uploads) > settings.pdf_max_files:
        raise PDFIngestionError('Select PDFs within the configured file-count limit.')
    total = 0
    names = set()
    for upload in uploads:
        name = upload.name
        if (not name or '/' in name or '\\' in name or ':' in name
                or Path(name).suffix.lower() != '.pdf' or name.endswith((' ', '.'))
                or any(ord(character) < 32 or character in '<>"|?*' for character in name)
                or name.split('.')[0].rstrip(' ').upper() in {
                    'CON', 'PRN', 'AUX', 'NUL',
                    *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
            raise PDFIngestionError('Each upload requires a plain PDF filename.')
        if name.casefold() in names:
            raise PDFIngestionError('Upload filenames must be distinct.')
        names.add(name.casefold())
        size = len(upload.content)
        total += size
        if not size or size > settings.pdf_max_file_bytes or total > settings.pdf_max_total_bytes:
            raise PDFIngestionError('Upload bytes exceed configured limits.')
        if not upload.content.startswith(b'%PDF-'):
            raise PDFIngestionError('File is not a recognizable PDF.')
    root = settings.pdf_upload_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='request-', dir=root) as directory:
        paths = []
        for upload in uploads:
            path = Path(directory) / upload.name
            path.write_bytes(upload.content)
            paths.append(path)
        yield paths


def generate(request: InterviewRequest | PresentationRequest, uploads: Sequence[Upload],
             settings: Settings, progress: Callable[[str], None] = lambda message: None) -> GenerationResult:
    """Prepare evidence once, then invoke the existing reviewed LangChain pipeline."""
    request = type(request).model_validate(request.model_dump())
    if isinstance(request, InterviewRequest) and request.question_count > settings.interview_max_questions:
        raise ValueError('Question count exceeds the configured limit.')
    document_mode = isinstance(request.source, DocumentInput)
    if not document_mode and uploads:
        raise ValueError('Online mode cannot include PDF uploads.')
    progress('Preparing configured AI providers')
    llm = get_llm(settings)
    if document_mode:
        progress('Extracting PDF text')
        with staged_pdfs(uploads, settings) as paths:
            loaded = load_pdfs(paths, settings)
        request = request.model_copy(update={'source': DocumentInput(document_ids=list(loaded.document_ids))})
        pages, warnings = loaded.pages, loaded.warnings
        urls = ()
    else:
        progress('Searching and fetching approved online sources')
        loaded = research_web(request, settings, llm, get_search_tool(settings), progress=progress)
        pages, warnings = loaded.pages, loaded.warnings
        urls = tuple(page.metadata['url'] for page in pages)
    progress('Splitting source text into searchable chunks')
    chunks = split_pages(pages, settings)
    progress('Loading the local embedding model (first run may download model files)')
    embeddings = get_embeddings(settings)
    progress('Embedding source chunks and building the local search index')
    store = (build_document_index(chunks, request.source.document_ids, embeddings)
             if document_mode else build_web_index(chunks, embeddings))
    retrieval_settings = settings
    if isinstance(request, PresentationRequest):
        k = settings.presentation_retrieval_k
        retrieval_settings = settings.model_copy(update={
            'retrieval_k': k, 'retrieval_fetch_k': max(k, settings.retrieval_fetch_k)})
    evidence_attempt = 0
    context_attempt = 0
    for attempt in range(settings.evidence_retry_limit + settings.context_retry_limit + 1):
        retriever = build_retriever(store, retrieval_settings)
        progress('Generating and reviewing the result')
        if isinstance(request, PresentationRequest):
            pipeline = PresentationPipeline(settings, retriever, build_presentation_chains(llm), web_urls=urls)
        elif document_mode:
            pipeline = PDFInterviewPipeline(settings, retriever, build_interview_chains(llm))
        else:
            pipeline = WebInterviewPipeline(settings, retriever, build_interview_chains(llm), list(urls))
        try:
            return GenerationResult(pipeline.run(request, progress=progress), tuple(warnings))
        except ContextLimitError:
            active_budget = (settings.presentation_context_max_chars if isinstance(request, PresentationRequest)
                             else settings.interview_context_max_chars)
            if context_attempt >= settings.context_retry_limit or active_budget <= 1500:
                raise
            context_attempt += 1
            progress('The provider rejected the context size; reducing retrieved evidence and retrying with your topic unchanged')
            settings = settings.model_copy(update={
                'interview_context_max_chars': max(1500, settings.interview_context_max_chars // 2),
                'presentation_context_max_chars': max(1500, settings.presentation_context_max_chars // 2),
            })
        except InsufficientEvidenceError as error:
            current_k = retrieval_settings.retrieval_k
            expanded_k = min(current_k * 2, settings.evidence_max_retrieval_k, len(chunks))
            if evidence_attempt >= settings.evidence_retry_limit or expanded_k <= current_k:
                # Trusted diagnostic only; never surface the LLM's raw reason or source content.
                error.recovery_attempts = evidence_attempt
                error.requested_output = request.output
                error.requested_count = request.question_count if isinstance(request, InterviewRequest) else 10
                raise
            evidence_attempt += 1
            progress(f'Not enough evidence in the first selection; retrieving up to {expanded_k} chunks from the same sources')
            retrieval_settings = retrieval_settings.model_copy(update={
                'retrieval_k': expanded_k,
                'retrieval_fetch_k': max(expanded_k, retrieval_settings.retrieval_fetch_k),
            })
    raise AssertionError('Unreachable evidence-recovery state')


def presentation_bytes(plan: PresentationPlan, settings: Settings,
                       progress: Callable[[str], None] = lambda message: None) -> bytes:
    """Render into an isolated directory; return bytes without persistent UI files."""
    root = PROJECT_ROOT / '.build' / 'ui-exports'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        config = settings.model_copy(update={'presentation_output_dir': Path(directory)})
        progress('Rendering editable slides and running PowerPoint layout/package checks')
        result = render_pptx(plan, 'presentation.pptx', config).read_bytes()
        progress('PowerPoint is ready to download')
        return result
