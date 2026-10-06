# Phase 11: editable PowerPoint export

This phase converts the reviewed `PresentationPlan` from Phase 10 into a ten-slide
PowerPoint. It runs locally without an LLM call, so a saved plan can be rendered
again without spending provider quota. It does not replace the document or web
retrieval and generation pipelines.

## Files and their responsibilities

- `src/export/pptx_renderer.py`: Python application boundary, Pydantic validation,
  filename and template checks, subprocess timeout and exclusive output creation.
- `src/export/render_pptx.mjs`: isolated JavaScript rendering adapter. It imports
  the reference deck, duplicates inspected layouts, replaces native text, adds
  citations to speaker notes and validates the exported package.
- `templates/reference.pptx`: your supplied Day 15 template. The original empty
  placeholder has been replaced with the actual reference deck.
- `templates/reference.json`: inspected layout/shape mappings, dimensions, fonts
  and SHA-256. The hash prevents silently rendering with a different template.
- `RenderingSettings` in `src/core/config.py`: local renderer configuration that
  loads independently of provider credentials.
- `tests/test_pptx_renderer.py`: offline tests for validation, timeouts, publication,
  preserving existing files and invoking the adapter without a shell.

## Why there is no new Python package

Pydantic, pathlib, subprocess, tempfile and zipfile handle the Python boundary.
Rendering uses the installed Node.js Artifact Tool runtime. LangChain still
handles retrieval and LLM generation; rendering is a deterministic export step.
No LangGraph or other agent framework is introduced.

The Artifact Tool dependency is supplied by this Codex installation. This is a
working local adapter, not yet a portable server deployment. Another machine needs
the compatible runtime and validator scripts, or a replacement rendering adapter.
Do not assume `pip install -r requirements.txt` installs this runtime.

## Run from the project root

First generate a reviewed JSON plan using the Phase 10 presentation CLI. Then:

```powershell
.venv\Scripts\python.exe -m src.export.pptx_renderer --plan outputs\day15-plan.json --output day15.pptx
```

The output appears inside `outputs`. Choose a new filename if it already exists.
The command rejects invalid schemas, missing templates, runtime failures and
detected text overflow. Shorten overflowing slide content rather than reducing
the template font sizes. Package/layout/font/import checks do not replace a human
review of factual accuracy and the deck's appearance in the target PowerPoint app.

## Runtime configuration

The local `.env` now contains verified runtime paths. `.env.example` documents:

```dotenv
RENDER_NODE_EXECUTABLE=node
RENDER_ARTIFACT_MODULE=
RENDER_SKILL_DIR=
RENDER_VALIDATION_PYTHON=python
RENDER_TIMEOUT_SECONDS=300
RENDER_MAX_PLAN_BYTES=2000000
```

Use absolute paths for the first four fields. `RENDER_ARTIFACT_MODULE` points to
the installed `@oai/artifact-tool/dist/artifact_tool.mjs`; `RENDER_SKILL_DIR` points
to the presentation skill directory containing `container_tools`. Use forward
slashes in quoted Windows `.env` paths to avoid dotenv escape sequences.

## Your learning task

Read `render_pptx()` first. Trace how a validated plan becomes an argument list,
why `shell=True` is avoided, and why the existing output is never overwritten.
Next compare one entry in `reference.json` with its layout branch in the adapter.
Render your own reviewed PDF or web plan, open it in PowerPoint, and inspect the
editable text, code and source notes. The renderer demo uses fixture content to
exercise layouts; it is not an LLM-reviewed teaching deck.

The next phase can add a user interface for source selection, multiple PDF
uploads, interview generation and presentation downloads.
