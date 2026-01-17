import json
import time
from typing import Any, Optional, Dict, List
from queue import Queue
from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_redis_client = None
_redis_available = None


def get_redis():
    global _redis_client, _redis_available
    
    if _redis_available is False:
        return None
    
    if _redis_client is not None:
        return _redis_client
    
    redis_url = Config.REDIS_URL
    if not redis_url:
        logger.info("REDIS_URL not configured - using in-memory storage")
        _redis_available = False
        return None
    
    try:
        import redis
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info(f"Redis connected successfully")
        _redis_available = True
        return _redis_client
    except ImportError:
        logger.warning("Redis package not installed - using in-memory storage")
        _redis_available = False
        return None
    except Exception as e:
        logger.warning(f"Redis connection failed: {e} - using in-memory storage")
        _redis_available = False
        return None


def is_redis_available() -> bool:
    return get_redis() is not None


class RedisCache:
    
    DEFAULT_TTL = {
        "price": Config.REDIS_PRICE_TTL_SECONDS,
        "route": Config.REDIS_ROUTE_TTL_SECONDS,
        "perp": Config.REDIS_PERP_TTL_SECONDS,
        "impact": Config.REDIS_ROUTE_TTL_SECONDS,
        "metrics": Config.REDIS_METRICS_TTL_SECONDS,
        "snapshot": 5,
        "paper_account": 3600,
        "portfolio_config": 86400,
    }
    
    _memory_cache: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def _get_memory_cache(cls, key: str) -> Optional[Any]:
        if key in cls._memory_cache:
            entry = cls._memory_cache[key]
            if time.time() < entry.get("expires", 0):
                return entry.get("value")
            else:
                del cls._memory_cache[key]
        return None
    
    @classmethod
    def _set_memory_cache(cls, key: str, value: Any, ttl: int):
        cls._memory_cache[key] = {
            "value": value,
            "expires": time.time() + ttl,
        }
    
    @classmethod
    def _delete_memory_cache(cls, key: str):
        if key in cls._memory_cache:
            del cls._memory_cache[key]
    
    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        redis_client = get_redis()
        
        if redis_client:
            try:
                value = redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug(f"Redis get error for {key}: {e}")
        
        return cls._get_memory_cache(key)
    
    @classmethod
    def set(cls, key: str, value: Any, ttl: Optional[int] = None, category: str = "price"):
        if ttl is None:
            ttl = cls.DEFAULT_TTL.get(category, 60)
        
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize value for key {key}: {e}")
            return False
        
        redis_client = get_redis()
        
        if redis_client:
            try:
                redis_client.setex(key, ttl, serialized)
                return True
            except Exception as e:
                logger.debug(f"Redis set error for {key}: {e}")
        
        cls._set_memory_cache(key, value, ttl)
        return True
    
    @classmethod
    def delete(cls, key: str) -> bool:
        redis_client = get_redis()
        
        if redis_client:
            try:
                redis_client.delete(key)
            except Exception as e:
                logger.debug(f"Redis delete error for {key}: {e}")
        
        cls._delete_memory_cache(key)
        return True
    
    @classmethod
    def get_price(cls, symbol: str) -> Optional[Dict[str, Any]]:
        return cls.get(f"price:{symbol}")
    
    @classmethod
    def set_price(cls, symbol: str, data: Dict[str, Any]):
        cls.set(f"price:{symbol}", data, category="price")
    
    @classmethod
    def get_route(cls, input_mint: str, output_mint: str) -> Optional[Dict[str, Any]]:
        return cls.get(f"route:{input_mint}_{output_mint}")
    
    @classmethod
    def set_route(cls, input_mint: str, output_mint: str, data: Dict[str, Any]):
        cls.set(f"route:{input_mint}_{output_mint}", data, category="route")
    
    @classmethod
    def get_perp_data(cls, market: str) -> Optional[Dict[str, Any]]:
        return cls.get(f"perp:{market}")
    
    @classmethod
    def set_perp_data(cls, market: str, data: Dict[str, Any]):
        cls.set(f"perp:{market}", data, category="perp")
    
    @classmethod
    def get_snapshot(cls) -> Optional[Dict[str, Any]]:
        return cls.get("snapshot:latest")
    
    @classmethod
    def set_snapshot(cls, data: Dict[str, Any]):
        cls.set("snapshot:latest", data, category="snapshot")
    
    @classmethod
    def get_paper_account(cls, session_id: str) -> Optional[Dict[str, Any]]:
        return cls.get(f"paper:{session_id}")
    
    @classmethod
    def set_paper_account(cls, session_id: str, data: Dict[str, Any]):
        cls.set(f"paper:{session_id}", data, category="paper_account")
    
    @classmethod
    def get_portfolio_config(cls, session_id: str) -> Optional[Dict[str, Any]]:
        return cls.get(f"portfolio:{session_id}")
    
    @classmethod
    def set_portfolio_config(cls, session_id: str, data: Dict[str, Any]):
        cls.set(f"portfolio:{session_id}", data, category="portfolio_config")
    
    @classmethod
    def clear_session(cls, session_id: str):
        cls.delete(f"paper:{session_id}")
        cls.delete(f"portfolio:{session_id}")
    
    @classmethod
    def get_impact_curve(cls, base: str, quote: str, side: str, max_size: float, points: int = 10) -> Optional[Dict[str, Any]]:
        key = f"impact:{base}_{quote}_{side}_{int(max_size)}_{points}"
        return cls.get(key)
    
    @classmethod
    def set_impact_curve(cls, base: str, quote: str, side: str, max_size: float, data: Dict[str, Any], points: int = 10):
        key = f"impact:{base}_{quote}_{side}_{int(max_size)}_{points}"
        cls.set(key, data, category="impact")
    
    @classmethod
    def get_venue_metrics(cls, venue: str) -> Optional[Dict[str, Any]]:
        return cls.get(f"metrics:venue:{venue}")
    
    @classmethod
    def set_venue_metrics(cls, venue: str, data: Dict[str, Any]):
        cls.set(f"metrics:venue:{venue}", data, category="metrics")
    
    @classmethod
    def get_hyperliquid_data(cls, symbol: str) -> Optional[Dict[str, Any]]:
        return cls.get(f"hyperliquid:{symbol}")
    
    @classmethod
    def set_hyperliquid_data(cls, symbol: str, data: Dict[str, Any]):
        cls.set(f"hyperliquid:{symbol}", data, category="perp")
    
    @classmethod
    def increment_counter(cls, key: str, amount: int = 1) -> int:
        redis_client = get_redis()
        
        if redis_client:
            try:
                return redis_client.incrby(key, amount)
            except Exception as e:
                logger.debug(f"Redis incr error for {key}: {e}")
        
        current = cls._get_memory_cache(key)
        new_val = (current or 0) + amount
        cls._set_memory_cache(key, new_val, 3600)
        return new_val
    
    @classmethod
    def add_to_list(cls, key: str, value: Any, max_len: int = 1000):
        redis_client = get_redis()
        
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError):
            return False
        
        if redis_client:
            try:
                redis_client.lpush(key, serialized)
                redis_client.ltrim(key, 0, max_len - 1)
                return True
            except Exception as e:
                logger.debug(f"Redis list error for {key}: {e}")
        
        current = cls._get_memory_cache(key) or []
        current.insert(0, value)
        cls._set_memory_cache(key, current[:max_len], 3600)
        return True
    
    @classmethod
    def get_list(cls, key: str, start: int = 0, end: int = -1) -> List[Any]:
        redis_client = get_redis()
        
        if redis_client:
            try:
                items = redis_client.lrange(key, start, end)
                return [json.loads(item) for item in items]
            except Exception as e:
                logger.debug(f"Redis list get error for {key}: {e}")
        
        current = cls._get_memory_cache(key) or []
        if end == -1:
            return current[start:]
        return current[start:end + 1]


class RedisPubSub:
    _subscribers: Dict[str, List[Queue]] = {}
    _lock = __import__('threading').Lock()
    
    @classmethod
    def publish(cls, channel: str, message: Dict[str, Any]):
        redis_client = get_redis()
        
        if redis_client:
            try:
                redis_client.publish(channel, json.dumps(message))
                return True
            except Exception as e:
                logger.debug(f"Redis publish error: {e}")
        
        with cls._lock:
            if channel in cls._subscribers:
                for queue in cls._subscribers[channel]:
                    try:
                        queue.put_nowait(message)
                    except:
                        pass
        return True
    
    @classmethod
    def subscribe(cls, channel: str, queue: 'Queue'):
        with cls._lock:
            if channel not in cls._subscribers:
                cls._subscribers[channel] = []
            cls._subscribers[channel].append(queue)
    
    @classmethod
    def unsubscribe(cls, channel: str, queue: 'Queue'):
        with cls._lock:
            if channel in cls._subscribers and queue in cls._subscribers[channel]:
                cls._subscribers[channel].remove(queue)
