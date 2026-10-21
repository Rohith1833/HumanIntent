import json
from typing import List, Union
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment or defaults."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = Field(
        default="llama3",
        validation_alias=AliasChoices("ollama_model", "OLLAMA_MODEL"),
        description="Default local Ollama model",
    )
    ollama_timeout: float = 120.0
    ollama_keep_alive: str = "15m"
    context_window_size: int = Field(
        default=8192,
        validation_alias=AliasChoices("context_window_size", "CONTEXT_WINDOW_SIZE"),
        description="Effective context window size of the LLM in tokens (e.g., 8192 for llama3)",
    )
    backend_cors_origins: Union[List[str], str] = ["http://localhost:3000"]

    # Phase 8 Persistence settings
    database_url: str = Field(
        default="sqlite:///./data/humanintent.db",
        validation_alias=AliasChoices("humanintent_database_url", "database_url"),
        description="SQLAlchemy database connection URL",
    )
    history_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("history_enabled", "humanintent_history_enabled"),
        description="Whether to persist prompt enhancement session history locally",
    )

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str) and v.startswith("["):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return v if isinstance(v, list) else ["http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
