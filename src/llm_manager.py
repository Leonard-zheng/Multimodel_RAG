"""
LLM (Large Language Model) instance management with singleton pattern
"""
from typing import Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from .utils import logger
from .config import settings
import threading
from dotenv import load_dotenv

load_dotenv()


class LLMManager:
    """Singleton manager for LLM instances to avoid costly recreation"""
    
    _instance = None
    _lock = threading.Lock()
    _llm_cache: dict[str, Any] = {}
    _embeddings_cache: dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_llm(self, 
                model_name: str = None, 
                temperature: float = None,
                provider: str = None) -> Any:
        """
        Get or create LLM instance with caching
        
        Args:
            model_name: Name of the model (None to use config default)
            temperature: Sampling temperature (None to use config default)
            provider: LLM provider ('google' or 'ollama')
            
        Returns:
            LLM instance
        """
        # Use config defaults if not specified
        if provider is None:
            provider = settings.provider
        if model_name is None:
            model_name = settings.llm_models[provider]
        if temperature is None:
            temperature = settings.llm_temperatures[provider]
            
        cache_key = f"{provider}:{model_name}:{temperature}"
        
        if cache_key not in self._llm_cache:
            with self._lock:
                if cache_key not in self._llm_cache:
                    logger.info(f"Creating new LLM instance: {cache_key}")
                    
                    if provider.lower() == "google":
                        self._llm_cache[cache_key] = ChatGoogleGenerativeAI(
                            model=model_name,
                            temperature=temperature
                        )
                    elif provider.lower() == "ollama":
                        self._llm_cache[cache_key] = ChatOllama(
                            model=model_name,
                            temperature=temperature
                        )
                    else:
                        raise ValueError(f"Unsupported LLM provider: {provider}")
                        
        else:
            logger.debug(f"Reusing cached LLM instance: {cache_key}")
            
        return self._llm_cache[cache_key]
    
    def get_embeddings(self, 
                      model_name: str = None,
                      provider: str = "google") -> Any:
        """
        Get or create embeddings instance with caching
        
        Args:
            model_name: Name of the embedding model (None to use config default)
            provider: Embedding provider (currently only 'google')
            
        Returns:
            Embeddings instance
        """
        # Use config default if not specified
        if model_name is None:
            model_name = settings.embedding_model_name
            
        cache_key = f"{provider}:{model_name}"
        
        if cache_key not in self._embeddings_cache:
            with self._lock:
                if cache_key not in self._embeddings_cache:
                    logger.info(f"Creating new embeddings instance: {cache_key}")
                    
                    if provider.lower() == "google":
                        self._embeddings_cache[cache_key] = GoogleGenerativeAIEmbeddings(
                            model=model_name
                        )
                    else:
                        raise ValueError(f"Unsupported embeddings provider: {provider}")
                        
        else:
            logger.debug(f"Reusing cached embeddings instance: {cache_key}")
            
        return self._embeddings_cache[cache_key]
    
    def clear_cache(self) -> None:
        """Clear all cached instances"""
        with self._lock:
            logger.info("Clearing LLM and embeddings cache")
            self._llm_cache.clear()
            self._embeddings_cache.clear()
    
    def get_cache_info(self) -> dict[str, int]:
        """Get information about cached instances"""
        return {
            "llm_instances": len(self._llm_cache),
            "embeddings_instances": len(self._embeddings_cache)
        }


# Global instance
llm_manager = LLMManager()
