"""Generate a PDF-grounded interview; this command sends evidence to LLM providers."""

import argparse
import sys

from src.chains.interview import build_interview_chains
from src.core.config import get_settings
from src.core.llm_factory import AllProvidersUnavailableError, get_llm
from src.ingestion.pdf_loader import load_pdfs
from src.export.interview import render_interview_markdown
from src.pipelines.interview import InsufficientEvidenceError, InterviewQualityError, PDFInterviewPipeline
from src.retrieval.chunking import split_pages
from src.retrieval.embeddings import get_embeddings
from src.retrieval.retriever import build_retriever
from src.retrieval.vector_store import build_document_index
from src.schemas.requests import Difficulty, DocumentInput, InterviewRequest, QuestionType


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PDF-grounded interview JSON using external LLM providers.")
    parser.add_argument('--pdf', nargs='+', required=True)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--level', choices=['junior', 'mid', 'senior'], required=True)
    parser.add_argument('--difficulty', choices=[d.value for d in Difficulty], required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--types', nargs='+', choices=[t.value for t in QuestionType], required=True)
    parser.add_argument('--domain')
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown')
    args = parser.parse_args()
    try:
        settings = get_settings()
        # Validate basic request fields before PDF extraction or model downloads.
        request = InterviewRequest(source=DocumentInput(document_ids=['pending']), topic=args.topic,
            job_role=args.role, experience_level=args.level, difficulty=args.difficulty,
            question_count=args.count, question_types=args.types, domain=args.domain)
        if request.question_count > settings.interview_max_questions:
            raise ValueError('Question count exceeds the configured limit.')
        loaded = load_pdfs(args.pdf, settings)
        request.source = DocumentInput(document_ids=list(loaded.document_ids))
        chunks = split_pages(loaded.pages, settings)
        index = build_document_index(chunks, loaded.document_ids, get_embeddings(settings))
        pipeline = PDFInterviewPipeline(settings, build_retriever(index, settings),
                                        build_interview_chains(get_llm(settings)))
        result = pipeline.run(request)
    except InsufficientEvidenceError:
        print('Insufficient PDF evidence. Reduce the count or provide more relevant material.', file=sys.stderr)
        return 2
    except InterviewQualityError:
        print('Interview did not pass validation. No unapproved output was returned.', file=sys.stderr)
        return 2
    except AllProvidersUnavailableError:
        print('All LLM providers are unavailable. Retry later.', file=sys.stderr)
        return 2
    except Exception as error:
        print(f'Interview generation failed ({type(error).__name__}). Check request, PDFs, dependencies and provider configuration.', file=sys.stderr)
        return 1
    for warning in loaded.warnings:
        print('Warning:', warning, file=sys.stderr)
    print(result.model_dump_json(indent=2) if args.format == 'json' else render_interview_markdown(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
