"""Validated application configuration.

No model clients, network requests, or directory creation happen here.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# config.py → core → src → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LLMProvider = Literal["groq", "gemini", "openrouter"]


class IngestionSettings(BaseSettings):
    """Local PDF processing can run without any LLM credentials."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8",
        extra="ignore", frozen=True, hide_input_in_errors=True,
    )
    app_name: str = Field(default="Questlume", min_length=1)
    pdf_upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    pdf_max_files: int = Field(default=10, ge=1)
    pdf_max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    pdf_max_total_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    pdf_max_pages_per_file: int = Field(default=300, ge=1)
    pdf_max_extracted_chars: int = Field(default=2_000_000, ge=1)
    chunk_size: int = Field(default=1_000, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)
    chunk_max_count: int = Field(default=20_000, ge=1)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", min_length=1)
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_local_files_only: bool = False
    retrieval_k: int = Field(default=5, ge=1)
    retrieval_fetch_k: int = Field(default=20, ge=1)
    retrieval_search_type: Literal["similarity", "mmr"] = "mmr"
    retrieval_mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_max_query_chars: int = Field(default=2_000, ge=1)

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.retrieval_fetch_k < self.retrieval_k:
            raise ValueError("RETRIEVAL_FETCH_K must be at least RETRIEVAL_K.")
        return self


class RenderingSettings(IngestionSettings):
    """Renderer settings can load without any generation API credentials."""

    render_node_executable: str = "node"
    render_artifact_module: str = ""
    render_skill_dir: str = ""
    render_validation_python: str = "python"
    render_template_dir: Path = PROJECT_ROOT / "templates"
    render_timeout_seconds: int = Field(default=300, ge=1)
    render_max_plan_bytes: int = Field(default=2_000_000, ge=1)
    presentation_output_dir: Path = PROJECT_ROOT / "outputs"


class Settings(RenderingSettings):
    """Configuration shared by the interview and presentation workflows."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    llm_provider: LLMProvider = "gemini"
    llm_fallback_providers: tuple[LLMProvider, ...] = ("groq", "openrouter")
    groq_model: str = ""
    groq_api_key: SecretStr | None = Field(default=None, repr=False)

    gemini_model: str = ""
    google_api_key: SecretStr | None = Field(default=None, repr=False)

    openrouter_model: str = ""
    openrouter_api_key: SecretStr | None = Field(default=None, repr=False)

    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    smoke_test_max_words: int = Field(default=120, ge=20, le=500)
    smoke_test_max_topic_chars: int = Field(default=500, ge=1, le=5_000)
    interview_max_questions: int = Field(default=10, ge=1)
    interview_revision_limit: int = Field(default=2, ge=0, le=5)
    interview_context_max_chars: int = Field(default=16_000, ge=1)
    evidence_retry_limit: int = Field(default=1, ge=0, le=2)
    evidence_max_retrieval_k: int = Field(default=40, ge=1)
    presentation_revision_limit: int = Field(default=2, ge=0, le=5)
    presentation_context_max_chars: int = Field(default=24_000, ge=1)
    presentation_retrieval_k: int = Field(default=12, ge=1)
    presentation_max_title_chars: int = Field(default=100, ge=1)
    presentation_max_bullets: int = Field(default=5, ge=1)
    presentation_max_bullet_chars: int = Field(default=220, ge=1)
    presentation_max_code_chars: int = Field(default=1_500, ge=1)
    presentation_max_notes_chars: int = Field(default=2_000, ge=1)
    web_search_provider: Literal["tavily", "google"] = "tavily"
    tavily_api_key: SecretStr | None = Field(default=None, repr=False)
    google_search_api_key: SecretStr | None = Field(default=None, repr=False)
    google_cse_id: str = ""
    web_query_count: int = Field(default=3, ge=1, le=5)
    web_results_per_query: int = Field(default=5, ge=1, le=10)
    web_max_pages: int = Field(default=8, ge=1)
    web_min_pages: int = Field(default=2, ge=1)
    web_timeout_seconds: float = Field(default=20, gt=0)
    web_max_redirects: int = Field(default=3, ge=0)
    web_max_page_bytes: int = Field(default=2_000_000, ge=1)
    web_max_page_chars: int = Field(default=50_000, ge=1)
    web_max_total_chars: int = Field(default=200_000, ge=1)
    web_allowed_domains: tuple[str, ...] = (
        "docs.python.org", "angular.dev", "learn.microsoft.com",
        "docs.langchain.com", "developer.mozilla.org",
    )

    @model_validator(mode="after")
    def validate_web_limits(self) -> Self:
        if self.web_min_pages > self.web_max_pages:
            raise ValueError("WEB_MIN_PAGES must not exceed WEB_MAX_PAGES.")
        return self


    @model_validator(mode="after")
    def validate_selected_provider(self) -> Self:
        """Require usable credentials and a model for the selected provider."""

        if not self.app_name.strip():
            raise ValueError("APP_NAME must not be blank.")

        providers = (self.llm_provider, *self.llm_fallback_providers)
        if len(set(providers)) != len(providers):
            raise ValueError("Primary and fallback providers must be unique.")
        for provider in providers:
            model = getattr(self, f"{provider}_model")
            key_name = "google_api_key" if provider == "gemini" else f"{provider}_api_key"
            key = getattr(self, key_name)
            if not model.strip() or model != model.strip():
                raise ValueError(f"{provider.upper()}_MODEL must be nonblank without surrounding whitespace.")
            if key is None:
                raise ValueError(f"{key_name.upper()} is required for configured providers.")
            value = key.get_secret_value()
            if not value.strip() or value != value.strip():
                raise ValueError(f"{key_name.upper()} must be nonblank without surrounding whitespace.")

        return self


@lru_cache(maxsize=1) #Avoids repeatedly reading and validating the same settings
def get_settings() -> Settings:
    """Load and validate settings once per application process."""

    return Settings()
