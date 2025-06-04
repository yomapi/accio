from typing import Optional, Any, Dict
from cachetools import TTLCache
from loguru import logger

from ..core.config import CacheConfig


class Cache:
    """Memory cache with TTL"""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache = TTLCache(maxsize=self.config.max_size, ttl=self.config.ttl)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            return self.cache.get(key)
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        """Set value in cache

        Args:
            key: Cache key
            value: Value to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cache[key] = value
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def clear(self):
        """Clear all cached items"""
        self.cache.clear()

    @property
    def size(self) -> int:
        """Current number of items in cache"""
        return len(self.cache)

    @property
    def maxsize(self) -> int:
        """Maximum cache size"""
        return self.cache.maxsize

    def __len__(self) -> int:
        return self.size
