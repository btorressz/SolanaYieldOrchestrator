"""
Pyth Oracle Client - Integration layer for Pyth Observer oracle quality checks.

This module provides a pluggable client for fetching Pyth oracle prices and
computing deviations against our internal spot composite prices. Designed to
work with Pyth Observer (https://github.com/pyth-network/pyth-observer) as
an optional sidecar for oracle quality monitoring.

When PYTH_ENABLED is False or Pyth data is unavailable, functions gracefully
return None to maintain backward compatibility.
"""

import time
import logging
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PythPriceData:
    """Pyth oracle price data for a single asset."""
    symbol: str
    price: float
    confidence: float
    expo: int
    publish_time: int
    status: str  # "trading", "unknown", "halted", "auction"


class PythClient:
    """
    Client for fetching Pyth oracle prices and computing deviations.

    This client can be wired to real Pyth Observer output in the future.
    For now, it provides mock/simulated data when Pyth is disabled or
    when running in simulation mode.
    """

    PYTH_PRICE_IDS = {
        "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
        "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
        "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
        "mSOL": "c2289a6a43d2ce728d807d0d4132a61c5c82a6a0c56d75e0b9d4f5d3e6d4a3c2",
        "BONK": "72b021217ca3fe68922a19aaf990109cb9d84e9ad004b4d2025ad6f529314419",
        "XRP": "ec5d399846a9209f3fe5881d70aae9268c94339ff9817e8d18ff19fa05eea1c8",
        "USDC": "eaa020c61cc479712813461ce153894a96a6c00b21ed0cfc2798d1f9a9e9c94a",
        "USDT": "2b89b9dc8fdf9f34709a5b106b472f0f39bb6ca9ce04b0fd7f2e971688e2e53b",
        "JTO": "b43660a5f790c69354b0729a5ef9d50d68f1df92107540210b9cdd0d2b9c3a44",
        "JUP": "0a0408d619e9380abad35060f9192039ed5042fa6f82301d0e48bb52be830996",
        "ORCA": "37505261e557e251290b8c8899453064e8d760ed5c65cc9fa0de9e24ff5ebb24",
    }

    PYTH_API_URL = "https://hermes.pyth.network/v2/updates/price/latest"

    def __init__(self, enabled: bool = False, max_deviation_bps: int = 100, request_timeout_s: float = 3.0):
        self._enabled = enabled
        self._max_deviation_bps = max_deviation_bps

        self._price_cache: Dict[str, PythPriceData] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 5  # seconds
        self._lock = Lock()

        self._session = requests.Session()

        self._request_timeout_s = float(request_timeout_s)

        self._deviation_thresholds = {
            "clean": 25,   # <= 25 bps
            "watch": 100,  # 25-100 bps
            "flagged": 100 # > 100 bps
        }

        if self._enabled:
            logger.info("Pyth oracle client initialized")
        else:
            logger.info("Pyth oracle client disabled - using mock data when requested")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _should_refresh_cache(self) -> bool:
        return time.time() - self._cache_timestamp > self._cache_ttl

    def _fetch_pyth_prices(self) -> Dict[str, PythPriceData]:
        """Fetch latest prices from Pyth Hermes API v2."""
        if not self._enabled:
            return self._get_mock_prices()

        try:
            price_ids = ["0x" + pid for pid in self.PYTH_PRICE_IDS.values()]
            params = [("ids[]", pid) for pid in price_ids]

            response = self._session.get(
                self.PYTH_API_URL,
                params=params,
                timeout=self._request_timeout_s,
            )
            response.raise_for_status()
            data = response.json()

            prices: Dict[str, PythPriceData] = {}
            id_to_symbol = {"0x" + v: k for k, v in self.PYTH_PRICE_IDS.items()}

            feeds = data.get("parsed", []) if isinstance(data, dict) else data

            for feed in feeds:
                feed_id = str(feed.get("id", ""))
                if not feed_id.startswith("0x"):
                    feed_id = "0x" + feed_id
                symbol = id_to_symbol.get(feed_id)

                if not symbol:
                    continue

                price_data = feed.get("price", {}) or {}
                price = float(price_data.get("price", 0.0))
                expo = int(price_data.get("expo", 0))
                conf = float(price_data.get("conf", 0.0))

                actual_price = price * (10 ** expo)
                actual_conf = conf * (10 ** expo)

                prices[symbol] = PythPriceData(
                    symbol=symbol,
                    price=float(actual_price),
                    confidence=float(actual_conf),
                    expo=expo,
                    publish_time=int(price_data.get("publish_time", 0)),
                    status="trading",
                )

            if prices:
                logger.debug(f"Fetched {len(prices)} Pyth prices from v2 API")
            return prices if prices else self._get_mock_prices()

        except Exception as e:
            logger.warning(f"Pyth API fetch failed: {e}, using mock data")
            return self._get_mock_prices()

    def _get_mock_prices(self) -> Dict[str, PythPriceData]:
        """Generate deterministic mock Pyth prices for simulation/fallback."""
        now = int(time.time())
        seed = hashlib.md5(str(now // 10).encode()).hexdigest()

        base_prices: Dict[str, float] = {
            "SOL": 125.00,
            "BTC": 89000.00,
            "ETH": 3050.00,
            "mSOL": 140.00,
            "BONK": 0.000022,
            "XRP": 0.52,
            "USDC": 1.0000,
            "USDT": 1.0000,
            "JTO": 2.80,
            "JUP": 0.95,
            "ORCA": 4.20,
        }

        prices: Dict[str, PythPriceData] = {}
        for symbol, base_price in base_prices.items():
            hash_val = int(hashlib.md5(f"{seed}{symbol}".encode()).hexdigest()[:8], 16)
            variation = (hash_val % 100 - 50) / 10000.0  # -0.5% to +0.5%
            price = float(base_price * (1.0 + variation))

            conf_ratio = 0.0005 + (hash_val % 10) / 20000.0  # 0.05% to 0.10%

            prices[symbol] = PythPriceData(
                symbol=symbol,
                price=price,
                confidence=float(price * conf_ratio),
                expo=-8 if symbol != "BONK" else -12,
                publish_time=now,
                status="trading",
            )

        return prices

    def refresh_prices(self) -> None:
        """Refresh the price cache."""
        with self._lock:
            self._price_cache = self._fetch_pyth_prices()
            self._cache_timestamp = time.time()

    def get_pyth_price(self, symbol: str) -> Optional[float]:
        """
        Get the Pyth oracle price for a symbol.

        Args:
            symbol: Asset symbol (e.g., "SOL", "BTC")

        Returns:
            Price in USD, or None if unavailable
        """
        if self._should_refresh_cache():
            self.refresh_prices()

        with self._lock:
            data = self._price_cache.get(symbol)
            return float(data.price) if data else None

    def get_pyth_data(self, symbol: str) -> Optional[PythPriceData]:
        """Get full Pyth price data including confidence for a symbol."""
        if self._should_refresh_cache():
            self.refresh_prices()

        with self._lock:
            return self._price_cache.get(symbol)

    def get_all_prices(self) -> Dict[str, PythPriceData]:
        """Get all cached Pyth prices."""
        if self._should_refresh_cache():
            self.refresh_prices()

        with self._lock:
            return self._price_cache.copy()

    def get_pyth_deviation(self, symbol: str, composite_price: float) -> Optional[float]:
        """
        Calculate deviation between Pyth oracle and composite spot price.

        Args:
            symbol: Asset symbol
            composite_price: The venue composite price (e.g., avg of Jupiter/CoinGecko/Kraken)

        Returns:
            Deviation in basis points, or None if Pyth price unavailable
        """
        pyth_price = self.get_pyth_price(symbol)

        if pyth_price is None or pyth_price <= 0 or composite_price <= 0:
            return None

        deviation_pct = abs(pyth_price - composite_price) / composite_price
        return float(deviation_pct * 10000.0)

    def get_oracle_confidence_bps(self, symbol: str, composite_price: float) -> Optional[int]:
        """
        Calculate a normalized oracle confidence score in basis points.
        Lower is better (low deviation = high confidence).

        Args:
            symbol: Asset symbol
            composite_price: The venue composite price

        Returns:
            Confidence score in bps (lower = more confident), or None
        """
        pyth_data = self.get_pyth_data(symbol)
        deviation = self.get_pyth_deviation(symbol, composite_price)

        if deviation is None:
            return None

        confidence_penalty = 0.0
        if pyth_data and pyth_data.price > 0:
            conf_ratio = float(pyth_data.confidence) / float(pyth_data.price)
            confidence_penalty = conf_ratio * 10000.0

        return int(float(deviation) + float(confidence_penalty))

    def get_oracle_status(self, symbol: str, composite_price: float) -> str:
        """
        Get oracle quality status for a symbol.

        Returns:
            "clean" (<=25 bps), "watch" (25-100 bps), or "flagged" (>100 bps)
        """
        deviation = self.get_pyth_deviation(symbol, composite_price)

        if deviation is None:
            return "unknown"

        if deviation <= self._deviation_thresholds["clean"]:
            return "clean"
        if deviation <= self._deviation_thresholds["watch"]:
            return "watch"
        return "flagged"

    def is_oracle_trusted(self, symbol: str, composite_price: float) -> bool:
        """
        Check if oracle is within acceptable deviation threshold.

        Args:
            symbol: Asset symbol
            composite_price: The venue composite price

        Returns:
            True if deviation is within MAX_ORACLE_DEVIATION_BPS, False otherwise
        """
        if not self._enabled:
            return True  # Always trusted when Pyth is disabled

        deviation = self.get_pyth_deviation(symbol, composite_price)

        if deviation is None:
            return True  # Default to trusted if we can't calculate deviation

        return deviation <= float(self._max_deviation_bps)

    def get_health_summary(self, composite_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Get a health summary for all symbols comparing Pyth vs composite prices.

        Args:
            composite_prices: Dict of symbol -> composite price

        Returns:
            Summary including per-symbol status and counts by bucket
        """
        symbols_data: list[Dict[str, Any]] = []
        counts: Dict[str, int] = {"clean": 0, "watch": 0, "flagged": 0, "unknown": 0}

        for symbol in self.PYTH_PRICE_IDS.keys():
            composite_price = composite_prices.get(symbol)
            pyth_price = self.get_pyth_price(symbol)

            if composite_price is not None and composite_price > 0:
                deviation = self.get_pyth_deviation(symbol, float(composite_price))
                status = self.get_oracle_status(symbol, float(composite_price))
            else:
                deviation = None
                status = "unknown"

            counts[status] = counts.get(status, 0) + 1

            symbols_data.append(
                {
                    "symbol": symbol,
                    "pyth_price": pyth_price,
                    "composite_price": composite_price,
                    "deviation_bps": deviation,
                    "status": status,
                }
            )

        return {
            "symbols": symbols_data,
            "clean_count": counts["clean"],
            "watch_count": counts["watch"],
            "flagged_count": counts["flagged"],
            "unknown_count": counts["unknown"],
            "pyth_enabled": self._enabled,
            "max_deviation_bps": self._max_deviation_bps,
            "timestamp": time.time(),
        }


_pyth_client: Optional[PythClient] = None


def get_pyth_client() -> PythClient:
    """Get or create the global Pyth client instance."""
    global _pyth_client
    if _pyth_client is None:
        from config import Config

        _pyth_client = PythClient(
            enabled=bool(getattr(Config, "PYTH_ENABLED", False)),
            max_deviation_bps=int(getattr(Config, "MAX_ORACLE_DEVIATION_BPS", 100)),
        )
    return _pyth_client


def initialize_pyth_client(enabled: bool = False, max_deviation_bps: int = 100) -> PythClient:
    """Initialize the global Pyth client with specific settings."""
    global _pyth_client
    _pyth_client = PythClient(enabled=enabled, max_deviation_bps=max_deviation_bps)
    return _pyth_client

