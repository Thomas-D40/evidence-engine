from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # Environment
    env: str = "development"

    # OpenAI — required, validated at startup
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_smart_model: str = "gpt-4o"

    # Logging
    log_level: str = "INFO"

    # API Security — empty means reject all (unlike video-analyzer where empty = open)
    allowed_api_keys: str = ""

    # Optional research API keys
    newsapi_key: Optional[str] = None
    gnews_api_key: Optional[str] = None
    google_factcheck_api_key: Optional[str] = None
    claimbuster_api_key: Optional[str] = None

    # Enrichment — smart full-text filtering
    fulltext_screening_enabled: bool = Field(
        default=True,
        description="Enable relevance screening before full-text fetch"
    )
    fulltext_top_n: int = Field(
        default=3,
        description="Number of top sources to fetch full text for"
    )
    fulltext_min_score: float = Field(
        default=0.6,
        description="Minimum relevance score (0.0-1.0) for full-text fetch"
    )

    # Adversarial query generation
    adversarial_queries_enabled: bool = Field(
        default=True,
        description="Enable adversarial query generation for dual-source retrieval"
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8001

    @model_validator(mode="after")
    def validate_openai_key(self) -> "Settings":
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required but not set. "
                "Set it in your .env file or as an environment variable."
            )
        return self

    @property
    def api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.allowed_api_keys.split(",") if k.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
