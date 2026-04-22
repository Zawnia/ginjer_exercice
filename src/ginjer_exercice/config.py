from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # GCP
    gcp_project_id: str | None = None
    google_application_credentials: str | None = None

    # LLM
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None

    # Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "http://localhost:3000"
    langfuse_enabled: bool = True

    # Pipeline
    fallback_confidence_threshold: float = 0.7
    web_verify_threshold: float = 0.85
    prompt_cache_ttl_seconds: int = 300

    # BigQuery
    bigquery_table: str = "ginjer-440122.ia_eng_interview.ads"

    # Results persistence
    sqlite_db_path: str = "data/results/pipeline_results.db"

    # Media fetcher
    media_max_size_bytes: int = 50 * 1024 * 1024  # 50 MB
    media_image_timeout: float = 30.0
    media_video_timeout: float = 120.0
    media_max_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    """Retourne l'instance singleton des paramètres de configuration."""
    return Settings()