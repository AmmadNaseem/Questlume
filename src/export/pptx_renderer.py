"""Python boundary for an isolated Artifact Tool PowerPoint renderer."""

import json
import os
import shutil
import re
from pathlib import Path
import subprocess
import tempfile
import zipfile

from src.core.config import PROJECT_ROOT, RenderingSettings
from src.schemas.presentation import PresentationPlan


class PresentationRenderError(RuntimeError):
    """Rendering or layout checks failed; no final presentation was delivered."""

    def __init__(self, message: str, *, slide_number: int | None = None,
                 text_box: str | None = None, reason: str = 'runtime'):
        super().__init__(message)
        self.slide_number = slide_number
        self.text_box = text_box
        self.reason = reason


def renderer_failure(stderr: str) -> PresentationRenderError:
    """Extract only known renderer diagnostics; never expose raw logs or content."""
    match = re.search(r'Slide (\d+): content does not fit (TextBox \d+); shorten the text\.', stderr)
    if match and 1 <= int(match[1]) <= 10:
        return PresentationRenderError('Slide text exceeds the available space.',
            slide_number=int(match[1]), text_box=match[2], reason='overflow')
    match = re.search(r'Slide (\d+): an unbroken word or code token exceeds its text frame\.', stderr)
    if match and 1 <= int(match[1]) <= 10:
        return PresentationRenderError('A word or code token exceeds the available width.',
            slide_number=int(match[1]), reason='width')
    if 'Template registry does not match the deck.' in stderr:
        return PresentationRenderError('The template was changed without updating its registry.', reason='template')
    if 'Cannot find module' in stderr or 'ERR_MODULE_NOT_FOUND' in stderr:
        return PresentationRenderError('A renderer runtime dependency is missing.', reason='dependency')
    return PresentationRenderError('Renderer rejected the deck. Check runtime or package/layout validation.')


def render_pptx(plan: PresentationPlan, filename: str,
                settings: RenderingSettings | None = None) -> Path:
    """Render validated slide content without calling an LLM or modifying templates.

    Runtime paths are configuration, not model-controlled values. Only registered
    template IDs and plain output filenames are accepted. Existing outputs survive.
    """
    config = settings if settings is not None else RenderingSettings()
    plan = PresentationPlan.model_validate(plan.model_dump())
    serialized = plan.model_dump_json()
    if len(serialized.encode('utf-8')) > config.render_max_plan_bytes:
        raise ValueError('Plan exceeds the configured size limit.')
    if Path(filename).name != filename or Path(filename).suffix.lower() != '.pptx':
        raise ValueError('Output must be a plain .pptx filename.')
    if not plan.template_id.isidentifier():
        raise ValueError('Template ID must be an identifier.')
    output = config.presentation_output_dir.resolve() / filename
    if output.exists():
        raise FileExistsError('Output already exists; choose a new filename.')
    registry = config.render_template_dir.resolve() / f'{plan.template_id}.json'
    if not registry.is_file():
        raise PresentationRenderError('Unknown template ID or missing registry.')
    try:
        template_name = json.loads(registry.read_text(encoding='utf-8'))['template_file']
    except (ValueError, KeyError, TypeError):
        raise PresentationRenderError('Invalid template registry.') from None
    if not isinstance(template_name, str) or Path(template_name).name != template_name:
        raise PresentationRenderError('Invalid template filename in registry.')
    template = registry.parent / template_name
    if template.resolve().parent != registry.parent.resolve():
        raise PresentationRenderError('Template must remain inside the template directory.')
    if not template.is_file() or not zipfile.is_zipfile(template):
        raise PresentationRenderError('Template is missing or is not a valid PPTX.')
    module = Path(config.render_artifact_module)
    skill = Path(config.render_skill_dir)
    if not config.render_artifact_module or not module.is_file():
        raise PresentationRenderError('Configure RENDER_ARTIFACT_MODULE for the installed Artifact Tool runtime.')
    if not config.render_skill_dir or not (skill / 'container_tools/artifact_tool_utils.mjs').is_file():
        raise PresentationRenderError('Configure RENDER_SKILL_DIR for presentation validation tools.')
    build_root = PROJECT_ROOT / '.build' / 'presentations'
    build_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=build_root) as directory:
        work = Path(directory)
        payload = work / 'plan.json'
        payload.write_text(serialized, encoding='utf-8')
        args = [config.render_node_executable, str(Path(__file__).with_name('render_pptx.mjs')),
                str(module.resolve()), str(skill.resolve()), str(payload), str(template),
                str(registry), str(work), str(PROJECT_ROOT)]
        try:
            process = subprocess.run(args, capture_output=True, text=True, encoding='utf-8',
                                     errors='replace', timeout=config.render_timeout_seconds,
                                     check=False, cwd=PROJECT_ROOT,
                                     env={**os.environ, 'RENDER_VALIDATION_PYTHON': config.render_validation_python})
        except subprocess.TimeoutExpired:
            raise PresentationRenderError('Rendering exceeded its configured time limit.', reason='timeout') from None
        except OSError:
            raise PresentationRenderError('Renderer could not start. Check the Node executable.', reason='dependency') from None
        if process.returncode:
            raise renderer_failure(process.stderr or '')
        candidate = work / 'final' / 'validated.pptx'
        if not candidate.is_file() or not zipfile.is_zipfile(candidate):
            raise PresentationRenderError('Renderer did not produce a validated PPTX.')
        # Exclusive publication prevents races from overwriting another result.
        with candidate.open('rb') as source:
            destination = output.open('xb')
            try:
                with destination:
                    shutil.copyfileobj(source, destination)
            except OSError:
                output.unlink(missing_ok=True)
                raise
        return output


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Render a 10-slide JSON plan into an editable PPTX.')
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True, help='New .pptx filename inside outputs.')
    args = parser.parse_args()
    try:
        settings = RenderingSettings()
        path = Path(args.plan)
        if path.stat().st_size > settings.render_max_plan_bytes:
            raise ValueError('Plan exceeds the configured size limit.')
        plan = PresentationPlan.model_validate_json(path.read_text(encoding='utf-8'))
        print(render_pptx(plan, args.output, settings))
    except Exception as error:
        print(f'Presentation rendering failed ({type(error).__name__}). Check JSON, template, runtime and content fit.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
