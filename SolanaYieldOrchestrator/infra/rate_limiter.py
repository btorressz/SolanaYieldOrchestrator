import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    requests_per_second: int
    requests_per_minute: int


class RateLimitExceeded(Exception):
    def __init__(self, venue: str, limit_type: str, current: int, max_allowed: int):
        self.venue = venue
        self.limit_type = limit_type
        self.current = current
        self.max_allowed = max_allowed
        super().__init__(
            f"Rate limit exceeded for {venue}: {limit_type} ({current}/{max_allowed})"
        )


class VenueRateLimiter:
    def __init__(self, venue: str, config: RateLimitConfig):
        self.venue = venue
        self.config = config
        self._lock = threading.Lock()
        
        self._second_window: deque = deque()
        self._minute_window: deque = deque()
    
    def _cleanup_windows(self, now: float):
        second_cutoff = now - 1.0
        while self._second_window and self._second_window[0] < second_cutoff:
            self._second_window.popleft()
        
        minute_cutoff = now - 60.0
        while self._minute_window and self._minute_window[0] < minute_cutoff:
            self._minute_window.popleft()
    
    def check_limit(self) -> bool:
        with self._lock:
            now = time.time()
            self._cleanup_windows(now)
            
            if len(self._second_window) >= self.config.requests_per_second:
                return False
            if len(self._minute_window) >= self.config.requests_per_minute:
                return False
            
            return True
    
    def acquire(self, raise_on_limit: bool = True) -> bool:
        with self._lock:
            now = time.time()
            self._cleanup_windows(now)
            
            if len(self._second_window) >= self.config.requests_per_second:
                if raise_on_limit:
                    raise RateLimitExceeded(
                        self.venue,
                        "per_second",
                        len(self._second_window),
                        self.config.requests_per_second
                    )
                return False
            
            if len(self._minute_window) >= self.config.requests_per_minute:
                if raise_on_limit:
                    raise RateLimitExceeded(
                        self.venue,
                        "per_minute",
                        len(self._minute_window),
                        self.config.requests_per_minute
                    )
                return False
            
            self._second_window.append(now)
            self._minute_window.append(now)
            return True
    
    def get_usage(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            self._cleanup_windows(now)
            
            return {
                "venue": self.venue,
                "per_second": {
                    "current": len(self._second_window),
                    "limit": self.config.requests_per_second,
                    "utilization": len(self._second_window) / self.config.requests_per_second
                },
                "per_minute": {
                    "current": len(self._minute_window),
                    "limit": self.config.requests_per_minute,
                    "utilization": len(self._minute_window) / self.config.requests_per_minute
                }
            }


class RateLimiterRegistry:
    _limiters: Dict[str, VenueRateLimiter] = {}
    _lock = threading.Lock()
    
    @classmethod
    def _get_config(cls, venue: str) -> RateLimitConfig:
        configs = {
            "jupiter": RateLimitConfig(
                Config.RATE_LIMIT_JUPITER_RPS,
                Config.RATE_LIMIT_JUPITER_RPM
            ),
            "drift": RateLimitConfig(
                Config.RATE_LIMIT_DRIFT_RPS,
                Config.RATE_LIMIT_DRIFT_RPM
            ),
            "hyperliquid": RateLimitConfig(
                Config.RATE_LIMIT_HYPERLIQUID_RPS,
                Config.RATE_LIMIT_HYPERLIQUID_RPM
            ),
            "coingecko": RateLimitConfig(5, 50),
            "kraken": RateLimitConfig(5, 100),
            "solana": RateLimitConfig(20, 600),
        }
        return configs.get(venue, RateLimitConfig(10, 300))
    
    @classmethod
    def get(cls, venue: str) -> VenueRateLimiter:
        with cls._lock:
            if venue not in cls._limiters:
                config = cls._get_config(venue)
                cls._limiters[venue] = VenueRateLimiter(venue, config)
            return cls._limiters[venue]
    
    @classmethod
    def acquire(cls, venue: str, raise_on_limit: bool = True) -> bool:
        limiter = cls.get(venue)
        return limiter.acquire(raise_on_limit)
    
    @classmethod
    def check(cls, venue: str) -> bool:
        limiter = cls.get(venue)
        return limiter.check_limit()
    
    @classmethod
    def get_all_usage(cls) -> Dict[str, Dict[str, Any]]:
        with cls._lock:
            return {
                venue: limiter.get_usage()
                for venue, limiter in cls._limiters.items()
            }
