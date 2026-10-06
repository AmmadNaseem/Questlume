"""Local embedding adapter, independent of generation providers."""

from langchain_core.embeddings import Embeddings

from src.core.config import IngestionSettings


def get_embeddings(settings: IngestionSettings | None = None) -> Embeddings:
    """Create once and reuse. The first use may download model weights.

    Text embedding is local. local_files_only=True requires pre-cached weights.
    Remote model code is disabled explicitly.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    config = settings if settings is not None else IngestionSettings()
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={
            "device": config.embedding_device,
            "trust_remote_code": False,
            "local_files_only": config.embedding_local_files_only,
        },
        encode_kwargs={"normalize_embeddings": True, "batch_size": config.embedding_batch_size},
    )
