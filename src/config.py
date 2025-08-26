"""
Simplified configuration management - only for core settings that need centralization
"""
from dataclasses import dataclass,field
from typing import Optional,Any


@dataclass
class Settings:
    """Core application settings - business configuration only"""
    
    # LLM Provider selection
    provider: str = "google"  # "google" or "ollama"
    
    # LLM settings by provider
    llm_models: dict[str, str] = field(default_factory=lambda: {
        "ollama": "llama3.1:8b",
        "google": "gemini-2.5-flash-lite"
    })
    
    llm_temperatures: dict[str, float] = field(default_factory=lambda: {
        "ollama": 0.0,
        "google": 0.0
    })

    
    # Embedding settings
    embedding_model_name: str = "models/gemini-embedding-001"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Default PDF path
    default_pdf_path: str = "./content/attention-is-all-you-need.pdf"


# Global settings instance
settings = Settings()