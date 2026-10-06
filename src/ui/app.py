"""Streamlit adapter: request collection, session-local results and downloads."""

import logging
from uuid import uuid4

import streamlit as st

from src.core.config import IngestionSettings, get_settings
from src.export.interview import render_interview_markdown
from src.schemas.presentation import PresentationPlan
from src.schemas.requests import Difficulty, DocumentInput, InterviewRequest, PresentationRequest, QuestionType, WebInput
from src.services.generation import Upload, generate, presentation_bytes
from src.ui.errors import UserInputError, error_message

logger = logging.getLogger(__name__)


def report_error(error: Exception) -> None:
    """Show actionable messages without disclosing keys, excerpts or raw traces."""
    reference = uuid4().hex[:12]
    logger.error('ui_operation_failed reference=%s error_type=%s', reference, type(error).__name__)
    st.error(error_message(error))
    st.caption(f'Error reference: {reference}. Share this reference if you need support.')


def show_result() -> None:
    result = st.session_state.get('generation_result')
    if result is None:
        return
    output = result.output
    st.subheader(f'Result: {output.topic}')
    st.caption(f'Source: {output.source_mode}. Downloads belong to this result, even if you change the form.')
    for warning in result.warnings:
        st.warning(warning)
    st.download_button('Download JSON', output.model_dump_json(indent=2),
                       file_name='presentation-plan.json' if isinstance(output, PresentationPlan) else 'interview.json',
                       mime='application/json', key='download_json')
    if isinstance(output, PresentationPlan):
        for slide in output.slides:
            with st.expander(f'{slide.number}. {slide.title}'):
                for bullet in slide.bullets:
                    st.write(bullet)
                if slide.code:
                    st.code(slide.code)
                if slide.speaker_notes:
                    st.write(slide.speaker_notes)
                st.caption('Evidence: '+', '.join(slide.source_ids))
        if st.button('Create PowerPoint', key='render_pptx'):
            try:
                with st.spinner('Rendering and validating 10 slides…'):
                    st.session_state['pptx_bytes'] = presentation_bytes(output, get_settings())
            except Exception as error:
                report_error(error)
        if st.session_state.get('pptx_bytes'):
            st.download_button('Download PowerPoint', st.session_state['pptx_bytes'],
                file_name=f'day-{output.day}.pptx',
                mime='application/vnd.openxmlformats-officedocument.presentationml.presentation', key='download_pptx')
    else:
        st.download_button('Download Markdown', render_interview_markdown(output),
                           file_name='interview.md', mime='text/markdown')
        for index, item in enumerate(output.items, 1):
            with st.expander(f'{index}. {item.question}'):
                st.write(item.answer)
                st.write('Key points')
                for point in item.key_points:
                    st.write(point)
                for follow_up in item.follow_up_questions:
                    st.write('Follow-up: '+follow_up)
                st.caption('Evidence: '+', '.join(item.source_ids))
    with st.expander('Sources used'):
        for source in output.sources:
            label = (f'{source.filename}, page {source.page}' if source.kind == 'document' else str(source.url))
            st.write(f'{source.source_id}: {label}')
            st.write(source.excerpt)


def main() -> None:
    # Startup stays usable without configured LLM keys; validate them on submission.
    try:
        limits = IngestionSettings()
    except Exception as error:
        report_error(error)
        return
    st.set_page_config(page_title=limits.app_name, page_icon='📚', layout='wide')
    st.title(limits.app_name)
    st.write('Generate interview questions and answers, or a 10-slide presentation, from your sources.')
    source_mode = st.radio('Knowledge source', ['Uploaded PDFs', 'Online research'], horizontal=True)
    output_mode = st.radio('Create', ['Interview questions', 'Presentation'], horizontal=True)
    with st.form('generation_form'):
        topic = st.text_input('Topic')
        files = []
        if source_mode == 'Uploaded PDFs':
            files = st.file_uploader('Upload one or more PDFs', type=['pdf'], accept_multiple_files=True)
            st.caption(f'Up to {limits.pdf_max_files} files; {limits.pdf_max_file_bytes // (1024*1024)} MB per file.')
        if output_mode == 'Interview questions':
            role = st.text_input('Job role', value='AI Engineer')
            level = st.selectbox('Experience level', ['junior', 'mid', 'senior'])
            difficulty = st.selectbox('Difficulty', [value.value for value in Difficulty])
            count = st.number_input('Number of questions', min_value=1, value=5, step=1)
            types = st.multiselect('Question types', [value.value for value in QuestionType], default=['conceptual'])
            domain = st.text_input('Technology / domain (optional)')
        else:
            day = st.number_input('Day', min_value=1, value=15, step=1)
            audience = st.text_input('Audience', value='AI engineering learners')
            st.caption('10 slides using your reference template.')
        st.caption('Generate sends selected source excerpts to your configured external LLM providers. Online mode also uses your search provider.')
        submitted = st.form_submit_button('Generate')
    if submitted:
        # A failed new request must not display an earlier result as its output.
        st.session_state.pop('generation_result', None)
        st.session_state.pop('pptx_bytes', None)
        progress = st.empty()
        try:
            source = DocumentInput(document_ids=['pending']) if source_mode == 'Uploaded PDFs' else WebInput()
            if output_mode == 'Interview questions':
                request = InterviewRequest(source=source, topic=topic, job_role=role,
                    experience_level=level, difficulty=difficulty, question_count=int(count),
                    question_types=types, domain=domain.strip() or None)
            else:
                request = PresentationRequest(source=source, topic=topic, day=int(day),
                    audience=audience, template_id='reference')
            if source_mode == 'Uploaded PDFs' and not files:
                raise UserInputError('Upload at least one PDF before generating, or choose Online research.')
            settings = get_settings()
            if output_mode == 'Interview questions' and count > settings.interview_max_questions:
                raise UserInputError(f'Request up to {settings.interview_max_questions} questions at a time. Reduce Number of questions and try again.')
            # Check reported upload sizes before copying bytes into application objects.
            if len(files) > settings.pdf_max_files:
                raise UserInputError(f'Upload up to {settings.pdf_max_files} PDFs at a time. Remove some files and try again.')
            if any(f.size > settings.pdf_max_file_bytes for f in files):
                raise UserInputError(f'Each PDF must be at most {settings.pdf_max_file_bytes / (1024*1024):g} MB. Compress or split the larger PDF.')
            if sum(f.size for f in files) > settings.pdf_max_total_bytes:
                raise UserInputError(f'The PDFs together must be at most {settings.pdf_max_total_bytes / (1024*1024):g} MB. Upload fewer files at a time.')
            uploads = [Upload(file.name, file.getvalue()) for file in files]
            with st.spinner('Preparing evidence and generating a reviewed result…'):
                st.session_state['generation_result'] = generate(request, uploads, settings, progress.info)
        except Exception as error:
            report_error(error)
        finally:
            progress.empty()
    show_result()
    if st.session_state.get('generation_result') is not None and st.button('Clear result'):
        st.session_state.pop('generation_result', None)
        st.session_state.pop('pptx_bytes', None)
        st.rerun()
