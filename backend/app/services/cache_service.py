"""
Cache Service - Redis-based caching for queries and embeddings
"""
import redis
import json
import hashlib
from typing import Optional, List, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self):
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis (optional - gracefully degrades if unavailable)"""
        try:
            # Parse Redis URL to handle authentication
            redis_url = settings.REDIS_URL
            
            # If Redis URL is not set or is placeholder, disable caching
            if not redis_url or redis_url == "redis://localhost:6379/0":
                logger.info("Redis not configured. Caching disabled (optional).")
                self.redis_client = None
                return
            
            # Connect with authentication support
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established - caching enabled")
            
        except redis.AuthenticationError as e:
            logger.warning(f"⚠️ Redis authentication failed: {e}. Check REDIS_URL credentials. Caching disabled.")
            self.redis_client = None
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Caching disabled (optional).")
            self.redis_client = None
        except Exception as e:
            logger.warning(f"⚠️ Redis error: {e}. Caching disabled (optional).")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, data: str) -> str:
        """Generate a cache key from data"""
        hash_obj = hashlib.md5(data.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get cached embedding for a query
        
        Args:
            query: The query text
            
        Returns:
            Cached embedding or None
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._generate_key("embedding", query)
            cached = self.redis_client.get(key)
            if cached:
                logger.info(f"Cache hit for query embedding")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set_query_embedding(self, query: str, embedding: List[float], ttl: int = 3600):
        """
        Cache an embedding for a query
        
        Args:
            query: The query text
            embedding: The embedding vector
            ttl: Time to live in seconds (default 1 hour)
        """
        if not self.redis_client:
            return
        
        try:
            key = self._generate_key("embedding", query)
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(embedding)
            )
            logger.info(f"Cached query embedding")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def get_query_results(self, query: str) -> Optional[Any]:
        """
        Get cached results for a query
        
        Args:
            query: The query text
            
        Returns:
            Cached results or None
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._generate_key("results", query)
            cached = self.redis_client.get(key)
            if cached:
                logger.info(f"Cache hit for query results")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set_query_results(self, query: str, results: Any, ttl: int = 1800):
        """
        Cache results for a query
        
        Args:
            query: The query text
            results: The results to cache
            ttl: Time to live in seconds (default 30 minutes)
        """
        if not self.redis_client:
            return
        
        try:
            key = self._generate_key("results", query)
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(results)
            )
            logger.info(f"Cached query results")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear_cache(self):
        """Clear all cache"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")


# Singleton instance
_cache_service = None


def get_cache_service() -> CacheService:
    """Get or create the cache service singleton"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
