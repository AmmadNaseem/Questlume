"""Generate a 10-slide content plan. Rendering to PPTX is a separate phase."""

import argparse
from pathlib import Path
import sys

from src.chains.presentation import build_presentation_chains
from src.core.config import get_settings
from src.core.llm_factory import get_llm
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.web_search import get_search_tool
from src.pipelines.presentation import PresentationPipeline
from src.pipelines.web_research import research_web
from src.retrieval.chunking import split_pages
from src.retrieval.embeddings import get_embeddings
from src.retrieval.retriever import build_retriever
from src.retrieval.vector_store import build_document_index, build_web_index
from src.schemas.requests import DocumentInput, PresentationRequest, WebInput


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded 10-slide JSON plan using external LLMs.")
    parser.add_argument('--source', choices=['document', 'web'], required=True)
    parser.add_argument('--pdf', nargs='+')
    parser.add_argument('--topic', required=True)
    parser.add_argument('--day', type=int, help='Optional positive day number; omit for an unnumbered presentation.')
    parser.add_argument('--audience', required=True)
    parser.add_argument('--template', required=True, help='Template ID, not a filesystem path.')
    parser.add_argument('--output', help='Optional new JSON filename inside outputs; existing files are preserved.')
    args = parser.parse_args()
    if args.source == 'document' and not args.pdf:
        parser.error('Document mode requires --pdf.')
    if args.source == 'web' and args.pdf:
        parser.error('Web mode cannot receive PDF files.')
    if args.output and (Path(args.output).name != args.output or Path(args.output).suffix.lower() != '.json'):
        parser.error('--output must be a plain .json filename.')
    try:
        settings = get_settings()
        if args.output and (settings.presentation_output_dir / args.output).exists():
            raise FileExistsError('Choose a new output filename; existing plans are preserved.')
        request = PresentationRequest(topic=args.topic, day=args.day, audience=args.audience,
            template_id=args.template, source=DocumentInput(document_ids=['pending']) if args.source == 'document' else WebInput())
        llm = get_llm(settings)
        if args.source == 'document':
            loaded = load_pdfs(args.pdf, settings)
            request.source = DocumentInput(document_ids=list(loaded.document_ids))
            chunks = split_pages(loaded.pages, settings)
            store = build_document_index(chunks, loaded.document_ids, get_embeddings(settings))
            urls = ()
        else:
            loaded = research_web(request, settings, llm, get_search_tool(settings))
            chunks = split_pages(loaded.pages, settings)
            store = build_web_index(chunks, get_embeddings(settings))
            urls = tuple(page.metadata['url'] for page in loaded.pages)
        retrieval_settings = settings.model_copy(update={
            'retrieval_k': settings.presentation_retrieval_k,
            'retrieval_fetch_k': max(settings.retrieval_fetch_k, settings.presentation_retrieval_k),
        })
        pipeline = PresentationPipeline(settings, build_retriever(store, retrieval_settings),
            build_presentation_chains(llm), web_urls=urls)
        plan = pipeline.run(request)
        payload = plan.model_dump_json(indent=2)
        if args.output:
            settings.presentation_output_dir.mkdir(parents=True, exist_ok=True)
            output = settings.presentation_output_dir / args.output
            with output.open('x', encoding='utf-8') as stream:
                stream.write(payload + '\n')
            print(f'Saved approved plan: {output}', file=sys.stderr)
    except Exception as error:
        print(f'Presentation planning failed ({type(error).__name__}). Check evidence, configuration, limits and output filename.', file=sys.stderr)
        return 2
    for warning in loaded.warnings:
        print('Warning:', warning, file=sys.stderr)
    print(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
