"""Safe, actionable UI messages. Raw provider errors and validation inputs stay private."""

from pydantic import ValidationError
from src.core.config import Settings
from src.core.llm_factory import AllProvidersUnavailableError, ContextLimitError
from src.export.pptx_renderer import PresentationRenderError
from src.ingestion.pdf_loader import PDFIngestionError, PDFTextUnavailableError
from src.ingestion.web_search import WebSearchError
from src.pipelines.interview import InsufficientEvidenceError, InterviewQualityError
from src.pipelines.presentation import PresentationQualityError
from src.pipelines.web_research import WebResearchError


class UserInputError(ValueError):
    """A UI-owned message built from trusted labels and numeric limits only."""


def error_message(error: Exception) -> str:
    if isinstance(error, UserInputError):
        return str(error)
    if isinstance(error, ContextLimitError):
        return 'The AI providers still rejected the context size after automatic reduction. Your topic was preserved. Ask the administrator to configure a model with suitable context/quota limits; shortening your topic is not required.'
    # SDK exceptions differ by provider; inspect bounded metadata, never raw response text.
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, 'status_code', None) or getattr(current, 'code', None)
        if status is None:
            status = getattr(getattr(current, 'response', None), 'status_code', None)
        if status == 401:
            return 'An AI or search provider rejected its API key. Ask the administrator to check that the key is valid and active; retrying alone will not fix this.'
        if status == 403:
            return 'A provider denied access to this request. Ask the administrator to check API permissions, model access and account restrictions.'
        if status == 404:
            return 'A configured model or provider endpoint was not found. Ask the administrator to check the model ID and whether it has been retired.'
        if status in (400, 413, 422):
            return 'A provider rejected the request format or size. Your topic does not need to change. Ask the administrator to check model compatibility and context limits using the error reference.'
        if status in (402, 429):
            return 'A provider has reached its quota or rate limit. Wait before retrying, or ask the administrator to check quota and fallback configuration.'
        if isinstance(status, int) and 500 <= status < 600:
            return 'A provider is experiencing a service error. Wait a few moments and retry.'
        current = current.__cause__ or current.__context__
    if isinstance(error, ImportError):
        return 'A required application dependency is missing or incompatible. Ask the administrator to install the project requirements in its virtual environment.'
    if isinstance(error, ValidationError):
        if error.title in ('Settings', 'IngestionSettings', 'RenderingSettings'):
            fields = {name.upper() for name in Settings.model_fields}
            messages = []
            for issue in error.errors(include_input=False):
                cause = str(issue.get('ctx', {}).get('error', ''))
                for field in fields:
                    if cause == f'{field} must be nonblank without surrounding whitespace.':
                        action = ('Enter the model ID' if field.endswith('_MODEL') else
                                  'Enter the API key' if field.endswith('_API_KEY') else 'Enter a value')
                        messages.append(f'{field} is missing or contains surrounding spaces. {action} in .env, remove extra spaces, then try again.')
                    elif cause == f'{field} is required for configured providers.':
                        messages.append(f'{field} is required. Ask the administrator to configure this provider in .env.')
                if issue['loc'] and str(issue['loc'][0]).upper() in fields:
                    field = str(issue['loc'][0]).upper()
                    messages.append(f'{field} has an invalid setting. Ask the administrator to correct it in .env.')
                if cause == 'Primary and fallback providers must be unique.':
                    messages.append('A model provider appears more than once. Ask the administrator to remove duplicates from the primary/fallback configuration.')
            return 'Configuration needs attention. '+ ' '.join(dict.fromkeys(messages or [
                'Ask the administrator to check model providers and configuration limits in .env.']))
        labels = {'topic': 'Topic', 'job_role': 'Job role', 'audience': 'Audience',
                  'question_count': 'Number of questions', 'question_types': 'Question types', 'day': 'Day'}
        messages = []
        for issue in error.errors(include_input=False):
            field = issue['loc'][0] if issue['loc'] else None
            if field not in labels:
                continue
            label = labels[field]
            if field == 'question_types':
                messages.append('Select at least one question type; each type should appear only once.')
            elif field in ('day', 'question_count'):
                messages.append(f'{label} must be a whole number greater than zero.')
            else:
                messages.append(f'{label} is required. Enter a value containing more than spaces.')
        return ' '.join(dict.fromkeys(messages)) or 'Some request fields are invalid. Check the required fields and selections, then try again.'
    if isinstance(error, PDFTextUnavailableError):
        return 'No readable text was found in a PDF. It may contain scanned images. Run OCR or upload a PDF with selectable text.'
    if isinstance(error, PDFIngestionError):
        known = {
            'Supply a nonempty sequence of PDF paths.': 'Upload at least one PDF before generating.',
            'Upload filenames must be distinct.': 'Two uploads have the same filename. Rename one file or remove the duplicate.',
            'Duplicate PDF content is not allowed.': 'The same PDF was uploaded more than once. Remove the duplicate and try again.',
            'Encrypted PDFs are not supported.': 'A PDF is password protected. Upload an unlocked copy that you are authorized to use.',
            'PDF page count exceeds the configured limit.': 'A PDF contains too many pages. Split it into smaller documents or ask the administrator to adjust the page limit.',
            'Extracted text exceeds the configured batch limit.': 'The documents contain too much text for one request. Select fewer PDFs or split them into smaller sections.',
            'File is not a recognizable PDF.': 'An uploaded file is not a valid PDF. Export it as a PDF; changing its filename extension is not enough.',
            'Each upload requires a plain PDF filename.': 'An upload has an unsupported filename. Rename it using a simple name ending in .pdf.',
            'Upload bytes exceed configured limits.': 'The upload exceeds the file or total batch size limit. Use smaller PDFs or fewer files.',
            'PDF size exceeds limits or the file is empty.': 'A PDF is empty or exceeds the file size limit. Upload a smaller, nonempty PDF.',
        }
        return known.get(str(error), 'A PDF could not be processed. Open it to check for corruption, then export a fresh, unlocked PDF and try again.')
    if isinstance(error, AllProvidersUnavailableError):
        return 'All configured AI providers are unavailable or have exhausted their quota. Wait and retry, or ask the administrator to check quotas and fallback providers.'
    if isinstance(error, InsufficientEvidenceError):
        retried = getattr(error, 'recovery_attempts', 0)
        prefix = 'We also searched a broader selection from the same sources. ' if retried else ''
        if getattr(error, 'requested_output', None) == 'presentation':
            return prefix+'The available evidence could not support 10 distinct slides. Choose a topic covered more fully by your sources, add relevant PDFs, or explicitly select Online research. No unsupported slides were generated.'
        return prefix+'The evidence could not support the requested questions at the chosen difficulty and types. Try 1–3 questions, a difficulty supported by the material, or conceptual/practical types. Add more relevant sources if needed. No unsupported answers were generated.'
    if isinstance(error, PresentationQualityError) and error.reason == 'format':
        return 'The AI repeatedly returned invalid JSON during slide drafting or review. Automatic format recovery could not validate the result. Ask the administrator to check structured-output model compatibility using this error reference.'
    if isinstance(error, PresentationQualityError) and error.reason == 'constraints':
        return 'The AI could not produce a slide plan with valid ordering, citations and content limits. No invalid deck was exported. Retry with concise slide content or a more specific topic.'
    if isinstance(error, PresentationQualityError):
        return 'The slide draft still contains unsupported or unsuitable content after automatic revisions. No unapproved deck was exported. Ask the administrator to inspect the presentation review diagnostics using this error reference.'
    if isinstance(error, InterviewQualityError):
        return 'The generated content did not pass its quality checks. No unapproved result was returned. Try a narrower topic or more relevant sources.'
    if isinstance(error, (WebResearchError, WebSearchError)):
        return 'Online research could not find enough usable sources. Try a more specific topic, or ask the administrator to check search credentials and allowed websites.'
    if isinstance(error, PresentationRenderError):
        if error.reason in ('overflow', 'width') and error.slide_number is not None:
            return (f'Slide {error.slide_number} contains text that does not fit its template. '
                    'Shorten its title, bullets or code lines, then render again. Your JSON plan is still available; '
                    'you do not need to regenerate the entire presentation.')
        if error.reason == 'template':
            return 'The reference template no longer matches its layout registry. Ask the administrator to restore the registered template or update its inspected mappings and hash.'
        if error.reason == 'dependency':
            return 'The PowerPoint runtime could not load a required component. Ask the administrator to check the Node executable and installed Artifact Tool dependencies. Your JSON plan remains available.'
        if error.reason == 'timeout':
            return 'PowerPoint rendering exceeded its time limit. Retry once; if it continues, ask the administrator to check renderer performance or increase RENDER_TIMEOUT_SECONDS.'
        if str(error).startswith('Configure RENDER_'):
            return 'PowerPoint export is not configured on this computer. Ask the administrator to configure the rendering runtime. You can still download your slide plan as JSON.'
        return 'PowerPoint export failed its runtime or layout checks. Keep your JSON plan, shorten long slide text, and ask the administrator to check the renderer if the problem continues.'
    if isinstance(error, (TimeoutError, ConnectionError)):
        return 'The service connection timed out or could not be established. Check connectivity and retry in a few moments.'
    if isinstance(error, ValueError) and str(error) == 'Question count exceeds the configured limit.':
        return 'Too many questions were requested. Reduce the question count to the configured maximum and try again.'
    return 'We could not complete this request. Try again once. If it keeps failing, share the error reference with the administrator.'
