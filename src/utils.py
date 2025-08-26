"""
Utility functions for error handling, logging, and common operations
"""
import logging
import os
import sys
from functools import wraps
from typing import Any, Callable, Optional

# Configure logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("multirag")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logging()

def handle_errors(operation: str = "operation"):
    """
    Decorator for handling common errors with logging
    
    Args:
        operation: Description of the operation being performed
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                logger.info(f"Starting {operation}")
                result = func(*args, **kwargs)
                logger.info(f"Successfully completed {operation}")
                return result
            except FileNotFoundError as e:
                logger.error(f"File not found during {operation}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error during {operation}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

def validate_file_path(file_path: str) -> None:
    """
    Validate that a file exists and is readable
    
    Args:
        file_path: Path to the file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")
    
    logger.info(f"File validation passed: {file_path}")

def validate_directory(dir_path: str, create_if_missing: bool = True) -> None:
    """
    Validate directory exists, optionally create it
    
    Args:
        dir_path: Directory path
        create_if_missing: Whether to create directory if it doesn't exist
        
    Raises:
        OSError: If directory cannot be created or accessed
    """
    if not os.path.exists(dir_path):
        if create_if_missing:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        else:
            raise OSError(f"Directory not found: {dir_path}")
    
    if not os.access(dir_path, os.W_OK):
        raise PermissionError(f"Directory is not writable: {dir_path}")
    
    logger.info(f"Directory validation passed: {dir_path}")

class MultiRagError(Exception):
    """Base exception class for MultiRag application"""
    pass

class DocumentProcessingError(MultiRagError):
    """Exception raised when document processing fails"""
    pass

class VectorStoreError(MultiRagError):
    """Exception raised when vector store operations fail"""
    pass

class RAGError(MultiRagError):
    """Exception raised when RAG operations fail"""
    pass