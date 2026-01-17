from __future__ import annotations

import time
import threading
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, cast

from config import Config
from infra.perp_venue import PerpVenue, VenueName, PerpPosition, PerpMarketData, OrderResult
from infra.metrics_tracker import metrics_tracker, LatencyTimer
from infra.rate_limiter import RateLimiterRegistry
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---- Optional Redis dependency (fix "RedisCache possibly unbound" + bad Literal typing) ----
try:
    from infra.redis_client import RedisCache as _RedisCache, is_redis_available as _is_redis_available

    REDIS_AVAILABLE = True
except Exception:
    _RedisCache = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False

    def _is_redis_available() -> bool:
        return False


def _redis_ok() -> bool:
    # Central guard so we never reference RedisCache when it doesn't exist.
    return bool(REDIS_AVAILABLE and _RedisCache is not None and _is_redis_available())


def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _get_slippage(kwargs: Dict[str, Any], default: float = 0.005) -> float:
    """
    Pyright-safe: always returns a float even if kwargs["slippage"] is Unknown/None.
    """
    raw = kwargs.get("slippage", default)
    if raw is None:
        return float(default)
    # If some type stubs mark kwargs.get as Unknown, we normalize via _to_float.
    return _to_float(raw, default)


@dataclass
class HyperliquidMarket:
    symbol: str
    sz_decimals: int
    max_leverage: int


class HyperliquidClient(PerpVenue):
    SYMBOL_MAP = {
        "SOL": "SOL",
        "BTC": "BTC",
        "ETH": "ETH",
        "SOL-PERP": "SOL",
        "BTC-PERP": "BTC",
        "ETH-PERP": "ETH",
    }

    def __init__(self):
        self._enabled = bool(Config.HYPERLIQUID_ENABLED)
        self._info_client = None
        self._exchange_client = None
        self._ws_manager = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False

        self._market_data: Dict[str, PerpMarketData] = {}
        self._positions: List[PerpPosition] = []
        self._account_value: float = 0.0
        self._lock = threading.Lock()

        self._last_update_ts: float = 0.0

        if self._enabled:
            self._initialize()
        else:
            self._load_mock_data()

    @property
    def name(self) -> VenueName:
        return VenueName.HYPERLIQUID

    def _initialize(self) -> None:
        try:
            try:
                from hyperliquid.info import Info
                from hyperliquid.exchange import Exchange
                from hyperliquid.utils import constants
            except ImportError as e:
                logger.warning(f"Hyperliquid SDK not available: {e}. Using mock data.")
                self._load_mock_data()
                return

            base_url = constants.TESTNET_API_URL if bool(Config.HYPERLIQUID_TESTNET) else constants.MAINNET_API_URL

            self._info_client = Info(base_url=base_url, skip_ws=True)

            if getattr(Config, "HYPERLIQUID_PRIVATE_KEY", None):
                try:
                    from eth_account import Account

                    account = Account.from_key(Config.HYPERLIQUID_PRIVATE_KEY)
                    self._exchange_client = Exchange(account, base_url=base_url)
                    logger.info("Hyperliquid Exchange client initialized")
                except ImportError:
                    logger.warning("eth_account not available, running in read-only mode")
            else:
                logger.info("Hyperliquid Info client initialized (read-only)")

            self._refresh_market_data()

        except Exception as e:
            logger.error(f"Failed to initialize Hyperliquid client: {e}")
            self._load_mock_data()

    def _load_mock_data(self) -> None:
        mock_markets = [
            ("SOL", 100.0, 0.0001, 87.6),
            ("BTC", 43000.0, 0.00005, 43.8),
            ("ETH", 2200.0, 0.00008, 70.08),
        ]

        for symbol, price, funding, apy in mock_markets:
            self._market_data[symbol] = PerpMarketData(
                symbol=symbol,
                mark_price=float(price),
                index_price=float(price) * 0.999,
                funding_rate=float(funding),
                funding_rate_apy=float(apy),
                open_interest=10_000_000.0,
                volume_24h=50_000_000.0,
                next_funding_time=time.time() + 3600,
            )

        self._last_update_ts = time.time()
        logger.info("Loaded mock Hyperliquid market data")

    def _refresh_market_data(self) -> None:
        if not self._info_client:
            return

        try:
            with LatencyTimer("hyperliquid", "get_meta"):
                RateLimiterRegistry.acquire("hyperliquid")
                meta = self._info_client.meta()

            with LatencyTimer("hyperliquid", "all_mids"):
                RateLimiterRegistry.acquire("hyperliquid")
                all_mids = self._info_client.all_mids()

            # hyperliquid SDK expects startTime (camelCase) in ms.
            funding_data: Dict[str, Optional[Dict[str, Any]]] = {}
            with LatencyTimer("hyperliquid", "funding_history"):
                RateLimiterRegistry.acquire("hyperliquid")
                start_ms = int(time.time() * 1000) - 3600000
                for symbol in ["SOL", "BTC", "ETH"]:
                    try:
                        fr = self._info_client.funding_history(symbol, startTime=start_ms)
                        funding_data[symbol] = fr[-1] if fr else None
                    except Exception:
                        funding_data[symbol] = None

            with self._lock:
                universe = (meta or {}).get("universe", []) or []
                for i, market in enumerate(universe):
                    symbol = str((market or {}).get("name", f"MARKET_{i}"))

                    mark_price = _to_float((all_mids or {}).get(symbol, 0.0), 0.0)

                    funding_rate = 0.0
                    fd = funding_data.get(symbol)
                    if fd:
                        funding_rate = _to_float(fd.get("fundingRate", 0.0), 0.0)

                    self._market_data[symbol] = PerpMarketData(
                        symbol=symbol,
                        mark_price=mark_price,
                        index_price=mark_price,
                        funding_rate=funding_rate,
                        funding_rate_apy=funding_rate * 24 * 365 * 100,
                        open_interest=0.0,
                        volume_24h=0.0,
                        next_funding_time=time.time() + 3600,
                    )

                self._last_update_ts = time.time()

        except Exception as e:
            logger.error(f"Failed to refresh Hyperliquid market data: {e}")

    def get_mark_price(self, symbol: str) -> Optional[float]:
        sym = self.SYMBOL_MAP.get(symbol, symbol)
        with self._lock:
            market = self._market_data.get(sym)
            return float(market.mark_price) if market else None

    def get_index_price(self, symbol: str) -> Optional[float]:
        sym = self.SYMBOL_MAP.get(symbol, symbol)
        with self._lock:
            market = self._market_data.get(sym)
            return float(market.index_price) if market else None

    def get_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = self.SYMBOL_MAP.get(symbol, symbol)
        with self._lock:
            market = self._market_data.get(sym)
            if not market:
                return None
            return {
                "rate": float(market.funding_rate),
                "apy": float(market.funding_rate_apy),
                "next_funding_time": float(market.next_funding_time),
            }

    def get_all_funding_rates(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for sym, market in self._market_data.items():
                result[sym] = {
                    "rate": float(market.funding_rate),
                    "apy": float(market.funding_rate_apy),
                    "next_funding_time": float(market.next_funding_time),
                }
        return result

    def get_market_data(self, symbol: str) -> Optional[PerpMarketData]:
        sym = self.SYMBOL_MAP.get(symbol, symbol)

        with self._lock:
            market = self._market_data.get(sym)

            if market and _redis_ok():
                try:
                    _RedisCache.set_hyperliquid_data(  # type: ignore[union-attr]
                        sym,
                        {
                            "symbol": market.symbol,
                            "mark_price": market.mark_price,
                            "index_price": market.index_price,
                            "funding_rate": market.funding_rate,
                            "funding_rate_apy": market.funding_rate_apy,
                            "open_interest": market.open_interest,
                            "volume_24h": market.volume_24h,
                            "next_funding_time": market.next_funding_time,
                        },
                    )
                except Exception:
                    pass

            return market

    def get_all_markets(self) -> List[PerpMarketData]:
        with self._lock:
            markets = list(self._market_data.values())

            if markets and _redis_ok():
                for market in markets:
                    try:
                        _RedisCache.set_hyperliquid_data(  # type: ignore[union-attr]
                            market.symbol,
                            {
                                "symbol": market.symbol,
                                "mark_price": market.mark_price,
                                "index_price": market.index_price,
                                "funding_rate": market.funding_rate,
                                "funding_rate_apy": market.funding_rate_apy,
                                "open_interest": market.open_interest,
                                "volume_24h": market.volume_24h,
                                "next_funding_time": market.next_funding_time,
                            },
                        )
                    except Exception:
                        pass

            return markets

    def open_position(
        self,
        symbol: str,
        size: float,
        side: str,
        reduce_only: bool = False,
        **kwargs,
    ) -> OrderResult:
        sym = self.SYMBOL_MAP.get(symbol, symbol)

        if Config.is_simulation() or not self._exchange_client:
            logger.info(f"[SIMULATION] Hyperliquid {side} {size} {sym}")
            return OrderResult(
                success=True,
                order_id=f"sim_{int(time.time() * 1000)}",
                simulated=True,
                venue="hyperliquid",
            )

        try:
            with LatencyTimer("hyperliquid", "place_order"):
                RateLimiterRegistry.acquire("hyperliquid")

                is_buy = side.lower() in ["long", "buy"]

                # FIX: pyright error "Unknown | None" -> always coerce to float
                slippage: float = _get_slippage(kwargs, default=0.005)  # 0.5% default

                # Keep these optional; SDK may accept None for tp/sl
                tp: Optional[float] = cast(Optional[float], kwargs.get("take_profit"))
                sl: Optional[float] = cast(Optional[float], kwargs.get("stop_loss"))

                order_result = self._exchange_client.market_open(  # type: ignore[call-arg]
                    sym,
                    is_buy,
                    float(size),
                    slippage,
                    tp,
                    sl,
                )

                if (order_result or {}).get("status") == "ok":
                    oid = (
                        (order_result or {})
                        .get("response", {})
                        .get("data", {})
                        .get("statuses", [{}])[0]
                        .get("resting", {})
                        .get("oid")
                    )
                    return OrderResult(success=True, order_id=str(oid), venue="hyperliquid")

                return OrderResult(success=False, error=str(order_result), venue="hyperliquid")

        except Exception as e:
            logger.error(f"Hyperliquid order failed: {e}")
            return OrderResult(success=False, error=str(e), venue="hyperliquid")

    def close_position(self, symbol: str, size: Optional[float] = None, **kwargs) -> OrderResult:
        sym = self.SYMBOL_MAP.get(symbol, symbol)

        if Config.is_simulation() or not self._exchange_client:
            logger.info(f"[SIMULATION] Hyperliquid close {sym}")
            return OrderResult(
                success=True,
                order_id=f"sim_close_{int(time.time() * 1000)}",
                simulated=True,
                venue="hyperliquid",
            )

        try:
            with LatencyTimer("hyperliquid", "market_close"):
                RateLimiterRegistry.acquire("hyperliquid")
                result = self._exchange_client.market_close(sym)  # type: ignore[call-arg]

                if (result or {}).get("status") == "ok":
                    return OrderResult(success=True, venue="hyperliquid")

                return OrderResult(success=False, error=str(result), venue="hyperliquid")

        except Exception as e:
            logger.error(f"Hyperliquid close failed: {e}")
            return OrderResult(success=False, error=str(e), venue="hyperliquid")

    def get_positions(self) -> List[PerpPosition]:
        if Config.is_simulation() or not self._info_client:
            return []

        try:
            with LatencyTimer("hyperliquid", "user_state"):
                RateLimiterRegistry.acquire("hyperliquid")

                from eth_account import Account

                account = Account.from_key(Config.HYPERLIQUID_PRIVATE_KEY)
                user_state = self._info_client.user_state(account.address)

            positions: List[PerpPosition] = []
            for pos in (user_state or {}).get("assetPositions", []) or []:
                position = (pos or {}).get("position", {}) or {}
                szi = _to_float(position.get("szi", 0.0), 0.0)
                if szi != 0.0:
                    positions.append(
                        PerpPosition(
                            symbol=str(position.get("coin", "")),
                            size=abs(szi),
                            side="long" if szi > 0 else "short",
                            entry_price=_to_float(position.get("entryPx", 0.0), 0.0),
                            mark_price=_to_float(position.get("markPx", 0.0), 0.0),
                            unrealized_pnl=_to_float(position.get("unrealizedPnl", 0.0), 0.0),
                            margin_used=_to_float(position.get("marginUsed", 0.0), 0.0),
                            leverage=_to_float(((position.get("leverage", {}) or {}).get("value", 1.0)), 1.0),
                            liquidation_price=_to_float(position.get("liquidationPx"), 0.0)
                            if position.get("liquidationPx") is not None
                            else None,
                        )
                    )

            return positions

        except Exception as e:
            logger.error(f"Failed to get Hyperliquid positions: {e}")
            return []

    def get_account_value(self) -> float:
        if Config.is_simulation() or not self._info_client:
            return 10000.0

        try:
            from eth_account import Account

            account = Account.from_key(Config.HYPERLIQUID_PRIVATE_KEY)
            user_state = self._info_client.user_state(account.address)
            return _to_float(((user_state or {}).get("marginSummary", {}) or {}).get("accountValue", 0.0), 0.0)

        except Exception as e:
            logger.error(f"Failed to get Hyperliquid account value: {e}")
            return 0.0

    def get_available_margin(self) -> float:
        if Config.is_simulation() or not self._info_client:
            return 10000.0

        try:
            from eth_account import Account

            account = Account.from_key(Config.HYPERLIQUID_PRIVATE_KEY)
            user_state = self._info_client.user_state(account.address)
            return _to_float((user_state or {}).get("withdrawable", 0.0), 0.0)

        except Exception as e:
            logger.error(f"Failed to get Hyperliquid margin: {e}")
            return 0.0

    def start_websocket(self) -> None:
        if (not self._enabled) or self._running:
            return

        self._running = True
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()
        logger.info("Hyperliquid WebSocket started")

    def _ws_loop(self) -> None:
        import websocket

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_ws_message(data)
            except Exception as e:
                logger.debug(f"WS message error: {e}")

        def on_error(ws, error):
            logger.error(f"Hyperliquid WS error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info("Hyperliquid WS closed")
            if self._running:
                time.sleep(5)
                self._connect_ws()

        def on_open(ws):
            logger.info("Hyperliquid WS connected")
            for symbol in ["SOL", "BTC", "ETH"]:
                ws.send(
                    json.dumps(
                        {
                            "method": "subscribe",
                            "subscription": {"type": "trades", "coin": symbol},
                        }
                    )
                )

        self._connect_ws = lambda: None  # type: ignore[assignment]

        while self._running:
            try:
                ws_url = str(Config.HYPERLIQUID_WS_URL)
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open,
                )
                self._connect_ws = lambda: ws.run_forever()  # type: ignore[assignment]
                ws.run_forever()
            except Exception as e:
                logger.error(f"WS connection error: {e}")
                time.sleep(5)

    def _handle_ws_message(self, data: Dict[str, Any]) -> None:
        channel = (data or {}).get("channel")

        if channel == "trades":
            trades = (data or {}).get("data", []) or []
            if trades:
                latest = trades[-1] or {}
                symbol = latest.get("coin")
                price = _to_float(latest.get("px", 0.0), 0.0)

                if symbol:
                    with self._lock:
                        if symbol in self._market_data:
                            self._market_data[symbol].mark_price = price
                            self._last_update_ts = time.time()

    def stop_websocket(self) -> None:
        self._running = False
        if self._ws_thread:
            self._ws_thread.join(timeout=2)

    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def get_health(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._enabled),
            "connected": bool(self._running),
            "last_update": float(self._last_update_ts),
            "markets_count": int(len(self._market_data)),
            "has_exchange_client": self._exchange_client is not None,
        }


hyperliquid_client = HyperliquidClient()
