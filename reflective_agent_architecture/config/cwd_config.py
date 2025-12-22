from pathlib import Path

import torch
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CWDConfig(BaseSettings):
    """
    Configuration for Cognitive Workspace Database.

    Loads from environment variables or .env file - never hardcode credentials!
    Set NEO4J_PASSWORD in your .env file or environment.

    Searches for .env file in:
    1. Current directory
    2. Parent directory (project root when running from src/)
    """

    # Find .env file in project root (one level up from src/)
    _env_file = Path(__file__).parent.parent.parent / ".env"

    model_config = SettingsConfigDict(
        env_file=str(_env_file) if _env_file.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(...)  # Required from environment - no default
    chroma_path: str = Field(default=str(Path(__file__).parent.parent.parent / "chroma_data"))
    # LLM Settings
    # LLM Settings
    llm_provider: str = Field(..., description="LLM provider (openrouter, ollama, etc.)")
    llm_model: str = Field(..., description="LLM model name")
    compass_model: str = Field(..., description="Model for COMPASS reasoning")

    # Ruminator Settings
    ruminator_enabled: bool = Field(default=False, description="Enable background rumination")
    ruminator_delay: float = Field(
        default=2.0, description="Delay between rumination cycles (seconds)"
    )

    # Embedding Settings
    embedding_provider: str = Field(
        ...,
        description="Embedding provider (sentence-transformers, ollama, lm_studio, openrouter)",
    )
    embedding_model: str = Field(..., description="Embedding model name")
    confidence_threshold: float = Field(default=0.7)  # Lower to .3 for asymmetric embeddings
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device for tensor operations",
    )

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "CWDConfig":
        """Ensure paths are absolute relative to project root."""
        if self.chroma_path and not Path(self.chroma_path).is_absolute():
            # Resolve relative to project root (parent of parent of this file)
            # This ensures consistent behavior regardless of CWD (e.g., running from src/ vs root)
            root = Path(__file__).parent.parent.parent
            self.chroma_path = str((root / self.chroma_path).resolve())
        return self
