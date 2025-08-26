"""
Cache management for summaries using stable content-based IDs
"""
import hashlib
import json
import os
import threading
from typing import Optional, Union
from .utils import logger


class CacheManager:
    """Manages local caching of summaries using stable content-based IDs"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "summaries.json")
        self._ensure_cache_dir()
        self._lock = threading.Lock()
        self._cache: dict[str, str] = self._load_cache()


    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {self.cache_dir}")


    def _load_cache(self) -> dict[str, str]:
        """Load existing cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                if not isinstance(cache, dict):
                    logger.warning("Cache file format invalid (expected object); starting with empty cache")
                    return {}
                # Ensure all keys/values are strings
                sanitized: dict[str, str] = {}
                for k, v in cache.items():
                    if isinstance(k, str) and isinstance(v, str):
                        sanitized[k] = v
                if len(sanitized) != len(cache):
                    logger.warning("Cache contained non-string keys/values; sanitized entries")
                cache = sanitized
                logger.info(f"Loaded {len(cache)} cached summaries")
                return cache
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file: {e}, starting with empty cache")
                return {}
        return {}


    def _save_cache(self) -> None:
        """Save cache to file"""
        tmp_path = f"{self.cache_file}.tmp"
        with self._lock:
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(self._cache, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.cache_file)  # atomic on POSIX/Windows
                logger.debug(f"Saved cache with {len(self._cache)} entries")
            except IOError as e:
                logger.error(f"Failed to save cache: {e}")
                # Best-effort cleanup of temp file
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass


    def generate_content_id(self, content: str) -> str:
        """Generate stable MD5-based ID from content"""
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        return hashlib.md5(content.encode('utf-8')).hexdigest()


    def get_summary(self, content_id: str) -> Optional[str]:
        """Get cached summary by content ID"""
        return self._cache.get(content_id)


    def set_summary(self, content_id: str, summary: str) -> None:
        """Cache a summary and save to disk"""
        self._cache[content_id] = summary
        self._save_cache()
        logger.debug(f"Cached summary for ID: {content_id[:8]}...")
    
    def has_summary(self, content_id: str) -> bool:
        """Check if summary exists in cache"""
        return content_id in self._cache
    
    def clear_cache(self) -> None:
        """Clear all cached summaries"""
        self._cache.clear()
        self._save_cache()
        logger.info("Cleared all cached summaries")
    
    def get_cache_stats(self) -> dict[str, Union[int, bool]]:
        """Get cache statistics"""
        return {
            "total_summaries": len(self._cache),
            "cache_file_exists": os.path.exists(self.cache_file)
        }

    def delete_summary(self, content_id: str) -> bool:
        """Delete a cached summary. Returns True if removed."""
        if content_id in self._cache:
            del self._cache[content_id]
            self._save_cache()
            logger.debug(f"Deleted cached summary for ID: {content_id[:8]}...")
            return True
        return False


# Global cache manager instance
cache_manager = CacheManager()
