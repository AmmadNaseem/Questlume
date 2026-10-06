"""Online interview CLI. Searches and sends fetched evidence to configured LLMs."""

import argparse
import sys

from src.chains.interview import build_interview_chains
from src.core.config import get_settings
from src.core.llm_factory import get_llm
from src.export.interview import render_interview_markdown
from src.ingestion.web_search import get_search_tool
from src.pipelines.interview import WebInterviewPipeline
from src.pipelines.web_research import research_web
from src.retrieval.chunking import split_pages
from src.retrieval.embeddings import get_embeddings
from src.retrieval.retriever import build_retriever
from src.retrieval.vector_store import build_web_index
from src.schemas.requests import Difficulty, InterviewRequest, QuestionType, WebInput


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate interviews from fetched online evidence.')
    parser.add_argument('--topic', required=True)
    parser.add_argument('--role', required=True)
    parser.add_argument('--level', choices=['junior', 'mid', 'senior'], required=True)
    parser.add_argument('--difficulty', choices=[d.value for d in Difficulty], required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--types', nargs='+', choices=[t.value for t in QuestionType], required=True)
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown')
    args = parser.parse_args()
    try:
        settings = get_settings()
        request = InterviewRequest(source=WebInput(), topic=args.topic, job_role=args.role,
            experience_level=args.level, difficulty=args.difficulty, question_count=args.count,
            question_types=args.types)
        if request.question_count > settings.interview_max_questions:
            raise ValueError('Question count exceeds the configured limit.')
        search = get_search_tool(settings)
        llm = get_llm(settings)
        research = research_web(request, settings, llm, search)
        chunks = split_pages(research.pages, settings)
        store = build_web_index(chunks, get_embeddings(settings))
        pipeline = WebInterviewPipeline(settings, build_retriever(store, settings),
            build_interview_chains(llm), [page.metadata['url'] for page in research.pages])
        result = pipeline.run(request)
    except Exception as error:
        print(f'Online generation failed ({type(error).__name__}). Check search keys, quotas, domains and model configuration.', file=sys.stderr)
        return 2
    for warning in research.warnings:
        print('Warning:', warning, file=sys.stderr)
    print(result.model_dump_json(indent=2) if args.format == 'json' else render_interview_markdown(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
