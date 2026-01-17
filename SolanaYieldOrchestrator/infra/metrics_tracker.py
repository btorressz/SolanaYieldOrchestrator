import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics

from utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from infra.redis_client import RedisCache, is_redis_available
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    def is_redis_available():
        return False


@dataclass
class LatencyRecord:
    venue: str
    operation: str
    latency_ms: float
    success: bool
    timestamp: float
    error: Optional[str] = None


@dataclass
class VenueMetrics:
    venue: str
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    failure_rate: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    last_error: Optional[str] = None
    last_error_timestamp: Optional[float] = None


@dataclass  
class MicroMetrics:
    timestamp: float
    pnl: float
    turnover: float
    inventory: Dict[str, float]
    fill_count: int


class MetricsTracker:
    def __init__(self, max_history: int = 1000, window_seconds: int = 300):
        self.max_history = max_history
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        
        self._latency_records: Dict[str, deque] = {}
        self._micro_metrics: deque = deque(maxlen=max_history)
        
        self._venue_stats: Dict[str, Dict[str, Any]] = {}
        
        self._pnl_history: deque = deque(maxlen=max_history)
        self._turnover_history: deque = deque(maxlen=max_history)
        self._inventory_history: deque = deque(maxlen=max_history)
    
    def record_latency(
        self,
        venue: str,
        operation: str,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        with self._lock:
            if venue not in self._latency_records:
                self._latency_records[venue] = deque(maxlen=self.max_history)
            
            record = LatencyRecord(
                venue=venue,
                operation=operation,
                latency_ms=latency_ms,
                success=success,
                timestamp=time.time(),
                error=error
            )
            self._latency_records[venue].append(record)
            
            if venue not in self._venue_stats:
                self._venue_stats[venue] = {
                    "total_requests": 0,
                    "failed_requests": 0,
                    "last_error": None,
                    "last_error_ts": None
                }
            
            self._venue_stats[venue]["total_requests"] += 1
            if not success:
                self._venue_stats[venue]["failed_requests"] += 1
                self._venue_stats[venue]["last_error"] = error
                self._venue_stats[venue]["last_error_ts"] = time.time()
    
    def get_venue_metrics(self, venue: str) -> VenueMetrics:
        with self._lock:
            records = self._latency_records.get(venue, [])
            stats = self._venue_stats.get(venue, {})
            
            cutoff = time.time() - self.window_seconds
            recent = [r for r in records if r.timestamp > cutoff]
            
            if not recent:
                return VenueMetrics(venue=venue)
            
            latencies = [r.latency_ms for r in recent if r.success]
            failures = [r for r in recent if not r.success]
            
            if latencies:
                sorted_lat = sorted(latencies)
                avg = statistics.mean(latencies)
                p50 = sorted_lat[len(sorted_lat) // 2]
                p95_idx = int(len(sorted_lat) * 0.95)
                p99_idx = int(len(sorted_lat) * 0.99)
                p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
                p99 = sorted_lat[min(p99_idx, len(sorted_lat) - 1)]
            else:
                avg = p50 = p95 = p99 = 0.0
            
            failure_rate = len(failures) / len(recent) if recent else 0.0
            
            return VenueMetrics(
                venue=venue,
                avg_latency_ms=round(avg, 2),
                p50_latency_ms=round(p50, 2),
                p95_latency_ms=round(p95, 2),
                p99_latency_ms=round(p99, 2),
                failure_rate=round(failure_rate, 4),
                total_requests=stats.get("total_requests", 0),
                failed_requests=stats.get("failed_requests", 0),
                last_error=stats.get("last_error"),
                last_error_timestamp=stats.get("last_error_ts")
            )
    
    def get_all_venue_metrics(self) -> Dict[str, VenueMetrics]:
        venues = list(self._latency_records.keys())
        return {venue: self.get_venue_metrics(venue) for venue in venues}
    
    def record_micro_metrics(
        self,
        pnl: float,
        turnover: float,
        inventory: Dict[str, float],
        fill_count: int = 0
    ):
        with self._lock:
            record = MicroMetrics(
                timestamp=time.time(),
                pnl=pnl,
                turnover=turnover,
                inventory=inventory.copy(),
                fill_count=fill_count
            )
            self._micro_metrics.append(record)
            
            self._pnl_history.append({"ts": record.timestamp, "value": pnl})
            self._turnover_history.append({"ts": record.timestamp, "value": turnover})
            self._inventory_history.append({"ts": record.timestamp, "value": inventory})
    
    def get_micro_metrics_timeseries(
        self,
        window_seconds: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            cutoff = time.time() - (window_seconds or self.window_seconds)
            
            pnl = [p for p in self._pnl_history if p["ts"] > cutoff]
            turnover = [t for t in self._turnover_history if t["ts"] > cutoff]
            inventory = [i for i in self._inventory_history if i["ts"] > cutoff]
            
            return {
                "pnl": pnl,
                "turnover": turnover,
                "inventory": inventory
            }
    
    def get_health_status(self, venue: str) -> Dict[str, Any]:
        metrics = self.get_venue_metrics(venue)
        
        if metrics.failure_rate > 0.5:
            status = "failing"
        elif metrics.failure_rate > 0.1 or metrics.p95_latency_ms > 2000:
            status = "degraded"
        else:
            status = "good"
        
        return {
            "venue": venue,
            "status": status,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "failure_rate": metrics.failure_rate,
            "last_error": metrics.last_error,
            "last_error_timestamp": metrics.last_error_timestamp
        }
    
    def record_api_call(self, venue: str):
        if REDIS_AVAILABLE and is_redis_available():
            try:
                minute_key = f"api_calls:{venue}:{int(time.time() // 60)}"
                RedisCache.increment_counter(minute_key, 1)
            except Exception:
                pass
        
        with self._lock:
            if venue not in self._venue_stats:
                self._venue_stats[venue] = {
                    "total_requests": 0,
                    "failed_requests": 0,
                    "last_error": None,
                    "last_error_ts": None
                }
            self._venue_stats[venue]["total_requests"] += 1
    
    def get_api_call_counts(self) -> Dict[str, Dict[str, Any]]:
        current_minute = int(time.time() // 60)
        venues = ["jupiter", "drift", "hyperliquid", "coingecko", "kraken", "pyth"]
        
        call_counts = {}
        for venue in venues:
            counts_per_minute = []
            total_recent = 0
            
            for i in range(5):
                minute_key = f"api_calls:{venue}:{current_minute - i}"
                count = 0
                if REDIS_AVAILABLE and is_redis_available():
                    try:
                        cached = RedisCache.get(minute_key)
                        if cached:
                            count = int(cached)
                    except Exception:
                        pass
                counts_per_minute.append(count)
                total_recent += count
            
            stats = self._venue_stats.get(venue, {})
            
            call_counts[venue] = {
                "calls_last_minute": counts_per_minute[0] if counts_per_minute else 0,
                "calls_last_5_minutes": total_recent,
                "total_requests": stats.get("total_requests", 0),
                "avg_calls_per_minute": round(total_recent / 5, 2)
            }
        
        return call_counts
    
    def get_hft_metrics(self) -> Dict[str, Any]:
        try:
            from infra.rate_limiter import RateLimiterRegistry
            rate_limits = RateLimiterRegistry.get_all_usage()
        except ImportError:
            rate_limits = {}
        
        venues_data = {}
        all_metrics = self.get_all_venue_metrics()
        api_call_counts = self.get_api_call_counts()
        
        all_venues = set(list(all_metrics.keys()) + list(api_call_counts.keys()))
        
        for venue in all_venues:
            metrics = all_metrics.get(venue)
            call_counts = api_call_counts.get(venue, {})
            rate_limit_data = rate_limits.get(venue, {})
            per_second = rate_limit_data.get("per_second", {})
            per_minute = rate_limit_data.get("per_minute", {})
            
            rate_limit_util = max(
                per_second.get("utilization", 0),
                per_minute.get("utilization", 0)
            ) if rate_limit_data else 0.0
            
            from config import Config
            venue_rps = getattr(Config, f"RATE_LIMIT_{venue.upper()}_RPS", 10)
            venue_rpm = getattr(Config, f"RATE_LIMIT_{venue.upper()}_RPM", 300)
            
            calls_per_minute = call_counts.get("calls_last_minute", 0)
            estimated_utilization = min(calls_per_minute / venue_rpm, 1.0) if venue_rpm > 0 else 0.0
            
            combined_utilization = max(rate_limit_util, estimated_utilization)
            
            venues_data[venue] = {
                "avg_latency_ms": metrics.avg_latency_ms if metrics else 0.0,
                "p50_latency_ms": metrics.p50_latency_ms if metrics else 0.0,
                "p95_latency_ms": metrics.p95_latency_ms if metrics else 0.0,
                "p99_latency_ms": metrics.p99_latency_ms if metrics else 0.0,
                "failure_rate": metrics.failure_rate if metrics else 0.0,
                "total_requests": metrics.total_requests if metrics else call_counts.get("total_requests", 0),
                "failed_requests": metrics.failed_requests if metrics else 0,
                "calls_last_minute": calls_per_minute,
                "calls_last_5_minutes": call_counts.get("calls_last_5_minutes", 0),
                "avg_calls_per_minute": call_counts.get("avg_calls_per_minute", 0.0),
                "rate_limit_rps": venue_rps,
                "rate_limit_rpm": venue_rpm,
                "rate_limit_utilization": round(combined_utilization, 4),
                "rate_limit_status": "near_limit" if combined_utilization > 0.8 else "ok",
                "last_error": metrics.last_error if metrics else None,
                "last_error_timestamp": metrics.last_error_timestamp if metrics else None
            }
        
        result = {
            "venues": venues_data,
            "rate_limits": rate_limits,
            "api_call_counts": api_call_counts,
            "timestamp": time.time(),
            "redis_backed": REDIS_AVAILABLE and is_redis_available()
        }
        
        if REDIS_AVAILABLE and is_redis_available():
            try:
                RedisCache.set_venue_metrics("hft_consolidated", result)
            except Exception:
                pass
        
        return result
    
    def persist_to_redis(self):
        if not REDIS_AVAILABLE or not is_redis_available():
            return False
        
        try:
            for venue in self._latency_records.keys():
                metrics = self.get_venue_metrics(venue)
                RedisCache.set_venue_metrics(venue, {
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "p50_latency_ms": metrics.p50_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "p99_latency_ms": metrics.p99_latency_ms,
                    "failure_rate": metrics.failure_rate,
                    "total_requests": metrics.total_requests,
                    "timestamp": time.time()
                })
            return True
        except Exception as e:
            logger.debug(f"Failed to persist metrics to Redis: {e}")
            return False


metrics_tracker = MetricsTracker()


class LatencyTimer:
    def __init__(self, venue: str, operation: str):
        self.venue = venue
        self.operation = operation
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            error = str(exc_val) if exc_val else None
            
            metrics_tracker.record_latency(
                venue=self.venue,
                operation=self.operation,
                latency_ms=latency_ms,
                success=success,
                error=error
            )
        
        return False
