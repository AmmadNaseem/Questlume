# Phase 12: user interface and application service

The Streamlit interface connects the existing source ingestion, retrieval,
reviewed interview/presentation pipelines and PPTX renderer. It does not add an
agent framework. LangChain continues to handle the LLM chains.

## Start the application

From the Questlume project directory:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Open the localhost URL printed by Streamlit. Configure your existing `.env` model
names and credentials before generating. Online mode also requires your search
provider's credentials and suitable `WEB_ALLOWED_DOMAINS`. The first local embedding
run may download the configured model. PowerPoint export needs the Phase 11 runtime.

## How to use it

1. Choose uploaded PDFs or online research.
2. Choose interview questions or a presentation.
3. Enter the topic and the output-specific fields. For PDF mode, upload one or
   several text-based PDFs; for slides, supply the day and audience.
4. Select Generate. Source excerpts are sent to configured external LLM providers.
5. Inspect answers/slides and their sources. Download JSON and interview Markdown.
6. For a presentation, select Create PowerPoint, then Download PowerPoint.

The slide JSON remains available if export fails. Changing fields does not
automatically call an LLM; generation runs only on form submission. A new failed
request clears the earlier displayed result so it cannot be mistaken for the new
output. Clear result removes the completed result and PPTX bytes from session state.

## What each file does and why

- `streamlit_app.py` is a small entry point. Running it from the project root makes
  the existing `src` imports work without creating a nested package.
- `src/ui/app.py` collects inputs, builds the existing Pydantic request contracts,
  calls the service, renders source references and offers downloads. Forms batch
  input changes. Session state retains a result across Streamlit reruns. Errors
  display safe messages and correlation references rather than raw traces or keys.
- `src/services/generation.py` owns the application use case. The UI does not need
  to know how FAISS, retrievers or provider fallbacks are wired. This separation
  allows a future FastAPI adapter to reuse the same generation function.
- `staged_pdfs()` validates count, bytes, names and PDF signatures before disk
  writes, creates a request-specific directory inside the upload root, and removes
  it after extraction, including on failure. Original filenames stay in citations.
- `generate()` reconstructs validated requests, selects exactly one source path,
  chunks/indexes the evidence, invokes the existing reviewed pipeline and returns
  a typed result plus ingestion warnings. Presentation retrieval uses its own k.
- `presentation_bytes()` renders in temporary storage and returns downloadable
  bytes. It does not leave generated UI decks in a shared output directory.
- `streamlit>=1.40,<2` is the only new direct package. It supplies forms, file
  uploads, progress messages, session state and downloads; it does not replace
  LangChain orchestration.

## Current scope

This is a local development interface with synchronous requests. Evidence indexes
and result objects are not globally cached or shared between browser sessions.
Browser/session downloads still contain document excerpts; treat them accordingly.
Uploads held by Streamlit remain until cleared or the session ends. PDF cleanup
is normal filesystem deletion, not a guarantee of secure erasure.

It is not yet a public enterprise deployment: authentication, tenant authorization,
background workers, concurrency/resource limits, retention policies and persisted
job status belong to the next production phases. Free-provider fallbacks remain
bounded and can all become unavailable; the UI reports that condition.

## Your learning task

Trace one PDF interview request from `st.form_submit_button` through `generate()`
to `PDFInterviewPipeline.run()`. Identify where Pydantic validation happens, where
text is extracted locally, and where evidence first goes to the external LLM.
Then trace online presentation generation and explain why its source scope is
different. Try a successful request and an invalid PDF, checking the download and
error behavior. Read the new offline service and UI tests before extending it.

Streamlit API references: https://docs.streamlit.io/develop/api-reference and
https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
