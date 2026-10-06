# Questlume

Questlume is a Python and LangChain application that turns selected knowledge
sources into interview preparation material or an editable ten-slide presentation.
Choose one or multiple uploaded PDFs, or research a topic online, then generate:

- Interview questions, detailed answers, key points, follow-up questions and source references.
- A reviewed presentation plan based on a topic, optional day and audience, followed by an
  editable PowerPoint using the supplied reference template.

This project is being developed incrementally for learning and extension toward
production. The current interface is a local Streamlit application. It is not yet
an authenticated, multi-tenant enterprise service.

## Contents

1. [Features and current scope](#features-and-current-scope)
2. [Setup and run commands](#setup-and-run-commands)
3. [Environment configuration](#environment-configuration)
4. [How to use the interface](#how-to-use-the-interface)
5. [Architecture and data flow](#architecture-and-data-flow)
6. [How the LLM layer works](#how-the-llm-layer-works)
7. [Project structure](#project-structure)
8. [Command-line workflows](#command-line-workflows)
9. [Limits and tuning](#limits-and-tuning)
10. [Testing and troubleshooting](#testing-and-troubleshooting)
11. [Privacy and production roadmap](#privacy-and-production-roadmap)

## Features and current scope

| Area | Implemented behavior |
| --- | --- |
| Document inputs | One or multiple text-based PDFs |
| Online inputs | Topic-based research using Tavily or Google search APIs and fetched pages |
| Interview parameters | Topic, job role, junior/mid/senior experience, difficulty, count, question types, optional domain |
| Question types | Conceptual, practical, scenario, coding, debugging, architecture, system design, best practices, security, performance, enterprise, behavioral |
| Presentation parameters | Topic, optional day, audience, exactly ten slides, reference template |
| LLM integrations | Groq, OpenRouter and Google Gemini, configurable order |
| Retrieval | Local Hugging Face embeddings and request-scoped, in-memory FAISS |
| Outputs | Interview JSON/Markdown, slide-plan JSON, editable PPTX |
| Validation | Pydantic contracts, citation/scope checks, bounded LLM review and revisions |
| Interface | Streamlit forms, multiple PDF uploads, progress, safe errors and downloads |

DOCX/TXT ingestion, OCR, Chroma, persistent indexes and public hosting are not
implemented. Uploaded documents and online sources are separate modes; a request
does not blend them. There is no LangGraph, CrewAI or AutoGen orchestration.

## Setup and run commands

### Prerequisites

- Python 3.11 or newer. The current development environment has been exercised
  with Python 3.13; dependency compatibility on another version should be checked.
- PowerShell on Windows for the commands below.
- Credentials and supported model IDs for every configured generation provider.
- A search API credential for online mode.
- Internet connectivity for APIs and the first embedding-model download.
- For PPTX export: the compatible Node.js Artifact Tool runtime and presentation
  validators described below. Interview generation and slide-plan generation do
  not require that renderer runtime.

### Create the environment

Run from the project folder. If `.venv` already exists and works, skip creating it.

```powershell
Set-Location 'E:\AamadNaseem\MSAIEngineering\AI Engineer\AiProjects\Questlume'
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
```

Explicitly using `.venv\Scripts\python.exe` prevents installing packages into a
different interpreter. Activation is optional:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Create configuration without overwriting existing credentials

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
notepad .env
```

Fill the required fields as explained below. Keep `.env` private and never put
real keys in `.env.example`, screenshots or source control. If a `.env` was already
tracked by Git, adding it to `.gitignore` does not remove that tracked copy.

### Start the user interface

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501` in your browser. Stop the process with **Ctrl+C** in
its terminal. To specify a different port:

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8502
```

Restart the server after changing `.env`: successfully loaded generation settings
are cached by `get_settings()`. A browser refresh alone does not reliably reload
that cache.

## Environment configuration

Settings are defined in `src/core/config.py` and read from the project-root `.env`.
Environment variables can override file values. List values use JSON syntax.
Generation configuration rejects unknown `.env` fields to catch misspellings.

### Generation providers

Example structure for all three providers:

```dotenv
APP_NAME=Questlume
LLM_PROVIDER=groq
LLM_FALLBACK_PROVIDERS=["openrouter","gemini"]

GROQ_MODEL=replace_with_a_supported_groq_model_id
GROQ_API_KEY=replace_with_your_groq_key
OPENROUTER_MODEL=replace_with_a_supported_openrouter_model_id
OPENROUTER_API_KEY=replace_with_your_openrouter_key
GEMINI_MODEL=replace_with_a_supported_gemini_model_id
GOOGLE_API_KEY=replace_with_your_gemini_key

LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2
```

These model/key values are placeholders, not working credentials. Copy supported
model IDs from your provider account; model availability and free access can
change. A blank model ID is a configuration failure even when its API key exists.
Remove leading/trailing spaces from values.

All providers listed in the primary/fallback order must have usable configuration.
If only Groq is configured while you set up the other accounts, use:

```dotenv
LLM_PROVIDER=groq
LLM_FALLBACK_PROVIDERS=[]
```

To use Gemini first:

```dotenv
LLM_PROVIDER=gemini
LLM_FALLBACK_PROVIDERS=["groq","openrouter"]
```

The primary provider must not also appear in the fallback list. Setting
`APP_NAME` controls the configured application name and UI title; package names,
documentation and local folder names remain repository identifiers.

### Online search

For Tavily:

```dotenv
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=replace_with_your_search_key
```

For Google search:

```dotenv
WEB_SEARCH_PROVIDER=google
GOOGLE_SEARCH_API_KEY=replace_with_your_search_key
GOOGLE_CSE_ID=replace_with_your_search_engine_id
```

`GOOGLE_SEARCH_API_KEY` is the search credential; `GOOGLE_API_KEY` is used by the
Gemini model integration. They are distinct application settings.

The default online allowlist is:

```dotenv
WEB_ALLOWED_DOMAINS=["docs.python.org","angular.dev","learn.microsoft.com","docs.langchain.com","developer.mozilla.org"]
WEB_QUERY_COUNT=3
WEB_RESULTS_PER_QUERY=5
WEB_MAX_PAGES=8
WEB_MIN_PAGES=2
```

Configure suitable authoritative domains for your topic. The current default is
technology-focused; topics such as general careers or business may require other
approved sources. The allowlist limits where fetching is permitted; it does not
prove that every statement on a page is correct.

### Local embeddings

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
EMBEDDING_LOCAL_FILES_ONLY=false
```

This embedding model converts text into vectors locally. It is separate from the
LLM that writes questions or slides. The first use may download model files. With
`EMBEDDING_LOCAL_FILES_ONLY=true`, those files must already be available locally.

### PowerPoint renderer

```dotenv
RENDER_NODE_EXECUTABLE="C:/path/to/node.exe"
RENDER_ARTIFACT_MODULE="C:/path/to/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
RENDER_SKILL_DIR="C:/path/to/presentations/skills/presentations"
RENDER_VALIDATION_PYTHON="C:/path/to/python.exe"
RENDER_TIMEOUT_SECONDS=300
RENDER_MAX_PLAN_BYTES=2000000
```

Use actual absolute paths and forward slashes in quoted Windows `.env` values.
The module path must refer to the installed Artifact Tool runtime; the skill
directory must contain its `container_tools` validators. The renderer also uses
the runtime's bundled `@napi-rs/canvas` for font-based content-fit checks.

**Portability limitation:** this adapter currently depends on resources bundled
with the development Codex installation. `pip install -r requirements.txt` does
not install these Node dependencies or validator scripts. Another computer/server
needs the compatible runtime or a replacement adapter. The Python application's
LLM orchestration remains independent of that rendering implementation.

## How to use the interface

### Interview questions from PDFs

1. Select **Uploaded PDFs** and **Interview questions**.
2. Upload one or several PDFs containing selectable text.
3. Enter a topic and job role; choose experience, difficulty, question count and types.
4. Optionally enter the technology/domain.
5. Select **Generate**. Relevant excerpts are sent to configured external LLMs.
6. Read the questions, answers, key points, follow-ups and evidence references.
7. Download the JSON or Markdown result.

### Interview questions from online research

1. Select **Online research** and **Interview questions**.
2. Enter the topic and interview parameters.
3. Select **Generate**. The app plans search queries, discovers allowed URLs,
   fetches pages and uses their actual text as evidence.
4. Inspect the source URLs and download the reviewed result.

### Presentation from PDFs or online research

1. Select your source mode and **Presentation**.
2. Enter the topic, optional day and audience. Upload PDFs when in document mode.
3. Select **Generate** to produce a reviewed ten-slide plan.
4. Inspect the slides and download JSON if you want to retain the plan.
5. Select **Create PowerPoint** to render the plan using `templates/reference.pptx`.
6. Select **Download PowerPoint** after rendering succeeds.

PowerPoint export is a separate local operation. It does not call an LLM and does
not consume generation quota. If it fails, the slide plan remains available.
Editing form fields does not automatically regenerate a result; press Generate
again. A failed new request clears the previous displayed result to avoid confusion.
Use **Clear result** to remove result/PPTX data from session state.

## Architecture and data flow

```text
Streamlit UI / CLI
        |
        v
Validated Pydantic request + configuration
        |
        +--- PDF mode: staged uploads -> validation -> PyPDFLoader text extraction
        |
        +--- Web mode: LLM search plan -> StructuredTool search -> approved URL fetch
        |
        v
Clean text + trusted source metadata
        |
        v
Text splitter -> local embeddings -> scoped in-memory FAISS -> retriever
        |
        v
Bounded evidence context
        |
        +--- Interview: question plan -> answer each question -> review each answer
        |                                      |
        |                                      v
        |                              Interview JSON / Markdown
        |
        +--- Presentation: slide draft -> content/schema/citation checks -> review
                                               |
                                               v
                                      Ten-slide JSON plan
                                               |
                                               v
                                  Template renderer -> checked editable PPTX
```

### Ingestion and source isolation

UI uploads are validated, saved under request-specific temporary directories in
`data/uploads`, extracted and then removed. Source records retain original
filenames and one-based page numbers. CLI PDF inputs must already be inside the
configured upload directory.

Web search snippets discover URLs; they are not used as grounding evidence. The
loader fetches approved HTTPS pages, checks redirects and public addresses, cleans
HTML and applies download/text limits. Fetching supports suitable HTML pages,
not arbitrary websites or remote PDF ingestion.

The service builds a FAISS index only for that request's selected source mode.
Document selection is filtered before embedding/indexing. Retrieved evidence is
checked against selected document IDs or fetched web URLs before generation.
Indexes are in memory and are not persisted or loaded from pickle files.

### Retrieval and RAG

RAG means retrieval-augmented generation: the LLM receives relevant source excerpts
instead of being asked to answer using its training knowledge alone. The splitter
preserves source metadata and stable chunk/reference IDs. MMR is the default
retrieval strategy; it balances relevance with diversity. Similarity retrieval is
also configurable.

Only retrieved, budgeted excerpts reach the LLM, not necessarily every page of a
large PDF. A document being within the byte limit does not mean all of its content
fits in one generation context. Tight topics and relevant documents improve
coverage. Grounding prompts, source-scope checks and review reduce unsupported
content, but do not mathematically guarantee factual correctness; review important
outputs against their citations.

### Interview generation

In the UI's application service, insufficient evidence triggers one automatic
recovery attempt by default: retrieval expands from the same request-scoped index
(normally 5 to 10 interview chunks or 12 to 24 presentation chunks). It reuses the
existing embeddings/index, keeps the original count and difficulty, and preserves
the context budget and quality review. It does not switch source modes. If there
are no additional chunks, it stops immediately. Recovery can add LLM calls.
`EVIDENCE_RETRY_LIMIT` controls additional attempts (default 1, maximum 2);
`EVIDENCE_MAX_RETRIEVAL_K` caps expanded retrieval (default 40). The direct CLI
pipeline calls do not currently use this application-service recovery loop.

The question-planning chain checks whether evidence can support the requested
count. The pipeline enforces count, unique questions, difficulty, selected types
and available source IDs. Each question receives additional question-specific
retrieval, then an answer and a grounding review. Changed question fields or
unavailable citations are rejected. Failed attempts can regenerate within the
configured revision limit. Unapproved output is not force-approved.

### Presentation generation and export

The draft chain produces ten structured slides, and the application supplies the
trusted source records. It checks slide order, cover placement, distinct titles,
citation membership and content-size limits before the review chain. The first
slide is the cover; ten slides includes that cover. Insufficient evidence returns
an error rather than filler slides.

The local renderer imports the supplied template, duplicates inspected layouts,
replaces editable text/code and adds citations/evidence to native speaker notes.
It checks the template hash, fonts, text fit, package structure, geometry and
round-trip import. It removes unused panels rather than leaving empty boxes.
These checks are not a substitute for inspecting the result in your target
PowerPoint application. CLI export protects existing output filenames; UI export
uses temporary storage and returns download bytes.

## How the LLM layer works

`src/core/llm_factory.py` creates provider-specific LangChain wrappers:

| Provider | Wrapper | Model field | Credential field |
| --- | --- | --- | --- |
| Groq | `ChatGroq` | `GROQ_MODEL` | `GROQ_API_KEY` |
| OpenRouter | `ChatOpenRouter` | `OPENROUTER_MODEL` | `OPENROUTER_API_KEY` |
| Gemini | `ChatGoogleGenerativeAI` | `GEMINI_MODEL` | `GOOGLE_API_KEY` |

Generation code consumes the resulting Runnable, not a particular provider's SDK.
Prompts are LangChain templates, with LCEL composition:

```text
PromptTemplate / ChatPromptTemplate | configured model Runnable | output parser
```

Structured generation chains use Pydantic output parsers. The model factory also
supports binding a Pydantic schema to each provider before composing fallbacks.
Plain Python control loops handle bounded revisions around Runnable invocation.
This is a staged workflow with specialized chains, not a team of autonomous agents
that independently chooses tools or coordinates through an agent framework.

### Calls made during a request

- PDF interview: planning, then answer generation and review for each question.
  Before revisions/fallbacks, a successful run normally needs `1 + 2 × count` LLM calls.
- Online interview: adds an LLM search-planning call and search/fetch API work.
- PDF presentation: slide drafting and review, normally two successful LLM calls.
- Online presentation: adds LLM search planning and search/fetch work.
- Embeddings: local model inference, separate from these LLM API calls.
- PPTX rendering: local export/validation only; no LLM calls.

Parsing failures, review rejection, SDK retries or provider fallbacks can increase
call counts and latency. Search planning itself may need retries.

### Fallback behavior

For `groq` with `["openrouter","gemini"]`, each eligible failed LLM invocation
tries Groq, then OpenRouter, then Gemini. The implementation classifies transport
errors, timeouts, HTTP 402/408/429 and HTTP 5xx as temporary/quota availability
failures and uses LangChain `with_fallbacks()` to route them.

Authentication errors, invalid requests and parsing/programming errors do not
automatically trigger provider fallback. Parser/review failures are handled by
the pipeline's bounded revision logic where applicable. If all providers are
unavailable, generation stops safely and reports a retry-later message. Free tiers
do not guarantee uninterrupted service. Fallback is per invocation; the app does
not permanently switch providers for the whole run or maintain a circuit breaker.

## Project structure

```text
Questlume/
├── streamlit_app.py          # UI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template, no real credentials
├── README.md
├── PHASE_11.md              # PowerPoint export learning guide
├── PHASE_12.md              # Interface/service learning guide
├── src/
│   ├── core/                # Configuration and provider abstraction
│   ├── schemas/             # Pydantic request, source and output contracts
│   ├── ingestion/           # PDF loading, search tools and safe web fetching
│   ├── retrieval/           # Chunking, embeddings, FAISS and retriever wiring
│   ├── prompts/             # Prompt templates kept separate from execution
│   ├── chains/              # LangChain prompt/model/parser Runnables
│   ├── pipelines/           # Interview, research, presentation and CLIs
│   ├── services/            # UI-independent application use cases
│   ├── export/              # Markdown output and isolated PPTX adapter
│   ├── ui/                  # Streamlit forms, display and safe error messages
│   └── main.py              # Minimal LCEL smoke-test CLI
├── templates/
│   ├── reference.pptx       # Supplied Day 15 reference deck
│   └── reference.json       # Inspected layouts, dimensions, fonts and hash
├── tests/                   # Offline unit and headless UI tests
├── data/uploads/            # CLI PDFs and temporary UI staging
├── outputs/                 # CLI plans/decks and local demo artifacts
└── .build/                  # Private temporary rendering/validation files
```

### Main dependency roles

| Package(s) | Why used |
| --- | --- |
| `langchain`, `langchain-core` | Runnable interfaces, LCEL, prompts, tools and parsers |
| `langchain-community` | PDF loader and FAISS integration |
| `langchain-text-splitters` | Chunking with overlap |
| `langchain-groq`, `langchain-openrouter`, `langchain-google-genai` | Swappable model integrations |
| `langchain-huggingface`, `sentence-transformers` | Local text embeddings |
| `faiss-cpu` | In-memory vector similarity search |
| `pydantic`, `pydantic-settings`, `python-dotenv` | Contracts and environment configuration |
| `pypdf` | PDF inspection and loader support |
| `httpx`, `beautifulsoup4` | Search/fetch requests and HTML cleaning |
| `streamlit` | Local forms, upload widgets, session state and downloads |

`requirements.txt` also contains libraries retained from earlier scaffolding,
including FastAPI/Uvicorn and additional helpers. Their presence does not mean a
public FastAPI API is implemented. Dependencies are not fully locked; deployment
reproducibility should be addressed before production.

## Command-line workflows

Run all commands from the project root. PowerShell backticks below continue lines;
do not put trailing spaces after a backtick. Replace filenames with your own.

### Offline chain wiring smoke test

```powershell
.venv\Scripts\python.exe -m src.main --topic "Python type hints" --offline
```

This uses a fixed fake response. It verifies basic wiring, not live AI generation.
Remove `--offline` to invoke the configured external providers.

### Load and chunk PDFs locally

Place CLI input files under `data/uploads` first:

```powershell
New-Item -ItemType Directory -Path data\uploads -Force
.venv\Scripts\python.exe -m src.ingestion.cli --pdf data\uploads\notes.pdf
```

This does not call an LLM or send PDF text externally.

### Search a PDF locally

```powershell
.venv\Scripts\python.exe -m src.retrieval.cli `
  --pdf data\uploads\notes.pdf `
  --query "How are type hints used?"
```

This may download embedding-model files but does not call generation providers.

### Generate PDF-grounded interviews

```powershell
.venv\Scripts\python.exe -m src.pipelines.cli `
  --pdf data\uploads\notes.pdf data\uploads\examples.pdf `
  --topic "Python type hints" --role "AI Engineer" --level senior `
  --difficulty advanced --count 5 --types conceptual scenario coding `
  --domain Python --format markdown
```

Use `--format json` for structured output. Both formats print to the console.

### Generate interviews using online research

```powershell
.venv\Scripts\python.exe -m src.pipelines.web_cli `
  --topic "Python type hints" --role "AI Engineer" --level senior `
  --difficulty advanced --count 5 --types conceptual practical --format json
```

### Generate a ten-slide PDF presentation plan

```powershell
.venv\Scripts\python.exe -m src.pipelines.presentation_cli `
  --source document --pdf data\uploads\notes.pdf `
  --topic "Python type hints" --day 15 --audience "AI engineering learners" `
  --template reference --output day15-plan.json
```

### Generate a ten-slide online presentation plan

```powershell
.venv\Scripts\python.exe -m src.pipelines.presentation_cli `
  --source web --topic "Python type hints" --day 15 `
  --audience "AI engineering learners" --template reference --output day15-web-plan.json
```

Omit `--day` when you do not want a day label. In the UI, leave **Include a day number** unchecked. The cover and headers then omit the day; the UI download is named `presentation.pptx`.

### Render an existing plan into an editable PPTX

```powershell
.venv\Scripts\python.exe -m src.export.pptx_renderer `
  --plan outputs\day15-plan.json --output day15.pptx
```

Plans/decks are written under the configured output directory. Use a new filename
when it already exists. Rendering an existing JSON plan does not require LLM API
credentials, but does require renderer configuration and a valid registered template.

## Limits and tuning

Defaults from the current configuration:

| Setting | Default | Meaning |
| --- | --- | --- |
| `PDF_MAX_FILES` | 10 | PDFs per request |
| `PDF_MAX_FILE_BYTES` | 20971520 | 20 MiB per PDF |
| `PDF_MAX_TOTAL_BYTES` | 104857600 | 100 MiB per batch |
| `PDF_MAX_PAGES_PER_FILE` | 300 | Pages per PDF |
| `PDF_MAX_EXTRACTED_CHARS` | 2000000 | Batch extracted-text budget |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Character-based chunking limits |
| `RETRIEVAL_K` / `RETRIEVAL_FETCH_K` | 5 / 20 | Selected candidates and MMR candidate pool |
| `INTERVIEW_MAX_QUESTIONS` | 10 | Maximum questions per request |
| `INTERVIEW_REVISION_LIMIT` | 2 | Revisions after the initial attempt |
| `INTERVIEW_CONTEXT_MAX_CHARS` | 16000 | Serialized evidence budget |
| `PRESENTATION_REVISION_LIMIT` | 2 | Slide-plan revisions after the initial attempt |
| `PRESENTATION_CONTEXT_MAX_CHARS` | 24000 | Serialized presentation evidence budget |
| `PRESENTATION_RETRIEVAL_K` | 12 | Presentation retrieval count |
| `PRESENTATION_MAX_TITLE_CHARS` | 100 | Slide title character cap |
| `PRESENTATION_MAX_BULLETS` | 5 | Bullets per slide |
| `PRESENTATION_MAX_BULLET_CHARS` | 220 | Characters per bullet |
| `PRESENTATION_MAX_CODE_CHARS` | 1500 | Code characters per slide |
| `PRESENTATION_MAX_NOTES_CHARS` | 2000 | Generated notes characters per slide |

Add supported uppercase settings to `.env` to override defaults. Chunk overlap
must be smaller than chunk size; retrieval fetch count must be at least retrieval
k; web minimum pages must not exceed maximum pages. Context limits are characters,
not token budgets. A provider's token/window constraints can still apply.

A 15 MB PDF is below the default byte limit, but can fail separate page/text limits,
encryption, corruption or missing-text checks. Raising limits increases CPU/RAM
use and request latency; it does not guarantee better grounding or model compatibility.

## Testing and troubleshooting

### Run the offline suite

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe -m pip check
```

Tests use fake/mocked providers for LLM and search work. They cover schemas,
fallback handling, ingestion, source isolation, retrieval, interview/presentation
review behavior, renderer boundaries, service routing and Streamlit interactions.
Passing them does not prove live credentials, model availability or factual quality.

### Common problems

| Symptom | What to do |
| --- | --- |
| Missing `GROQ_MODEL` or another model setting | Enter a supported model ID without surrounding spaces, then restart the app |
| Missing fallback credentials | Configure every listed provider, or temporarily remove unconfigured providers from the fallback list |
| Settings did not change after editing `.env` | Stop/restart Streamlit; successful settings are cached |
| Required field error | Enter Topic/job role/audience as appropriate; choose at least one question type |
| Too many questions | Reduce the count to `INTERVIEW_MAX_QUESTIONS` or intentionally tune that setting |
| Upload rejected | Check per-file/batch limits, page count, duplicate names/content and valid PDF format |
| No readable text | Use a PDF with selectable text or apply OCR outside the current app |
| Too few online sources | Check search credentials, allowed domains and topic specificity |
| All models unavailable | Wait, check quotas/connectivity, or configure available fallback models |
| Poor-quality/unsupported output | Narrow the topic, add stronger evidence or reduce question count |
| Embedding download failed | Check connectivity/cache and `EMBEDDING_LOCAL_FILES_ONLY` |
| PPTX runtime not configured | Set the absolute runtime/validator paths; JSON plans remain downloadable |
| PPTX layout check fails | Shorten slide text/code and check template/runtime configuration |
| Existing output filename | Choose a new JSON/PPTX filename; CLI output is not overwritten |

UI errors explain recovery steps and include a correlation reference. Logs record
the reference and error type without printing raw credentials or full provider
payloads. This is basic diagnostics, not a complete persisted observability system.

## Privacy and production roadmap

PDF extraction, embeddings and FAISS retrieval run locally. Selected evidence
excerpts, requests and generated candidates are sent to configured LLM providers
for drafting and review. With fallbacks enabled, evidence can reach multiple
providers. Online queries also go to the configured search service.

Treat source excerpts, output JSON, browser downloads and slide speaker notes as
potentially sensitive. UI staging files are deleted after extraction; normal file
deletion is not secure erasure. Streamlit retains uploaded bytes/session results
until cleared or the session ends. CLI input files and saved output files remain
on disk. The UI does not globally cache source documents or vector stores.

Before public deployment, add and verify:

- Authentication, tenant ownership checks and role-based access.
- Background jobs, cancellation, concurrency limits and persistent job status.
- Provider circuit breakers, quota visibility and explicit operational policies.
- Token-aware context budgeting and broader document ingestion/OCR.
- Retrieval/answer evaluation datasets and quality regression checks.
- Structured persisted logging, tracing, metrics and retention controls.
- Safe tenant-scoped caching and storage lifecycle policies.
- Locked dependencies, deployment configuration and a portable renderer adapter.

For a learning walkthrough, start with `src/schemas/requests.py`, then follow
`src/services/generation.py` into ingestion, retrieval and the relevant pipeline.
See [Phase 11](PHASE_11.md) for rendering and [Phase 12](PHASE_12.md) for UI/service
responsibilities and practice tasks.
