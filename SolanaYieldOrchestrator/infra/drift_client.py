from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Protocol, runtime_checkable, cast

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


# -----------------------------
# Optional dependency strategy
# -----------------------------
# These driftpy imports commonly fail in environments where driftpy isn't installed.
# Fix: make them optional + provide Protocol stubs so pyright stops complaining.
try:
    from driftpy.drift_client import DriftClient as DriftPyClient  # type: ignore[import-not-found]
    from driftpy.account_subscription_config import AccountSubscriptionConfig  # type: ignore[import-not-found]
    from driftpy.keypair import load_keypair  # type: ignore[import-not-found]
    from driftpy.types import (  # type: ignore[import-not-found]
        PositionDirection as DriftDirection,
        OrderParams,
        OrderType,
        MarketType,
    )

    DRIFTPY_AVAILABLE = True
except Exception:  # ImportError + other module resolution errors
    DriftPyClient = None  # type: ignore[assignment]
    AccountSubscriptionConfig = None  # type: ignore[assignment]
    load_keypair = None  # type: ignore[assignment]
    DriftDirection = None  # type: ignore[assignment]
    OrderParams = None  # type: ignore[assignment]
    OrderType = None  # type: ignore[assignment]
    MarketType = None  # type: ignore[assignment]
    DRIFTPY_AVAILABLE = False


@runtime_checkable
class DriftUserLike(Protocol):
    def get_total_collateral(self) -> int: ...
    def get_perp_positions(self) -> List[Any]: ...


@runtime_checkable
class DriftClientLike(Protocol):
    async def subscribe(self) -> None: ...
    async def unsubscribe(self) -> None: ...
    def get_user(self) -> Optional[DriftUserLike]: ...
    async def place_perp_order(self, order_params: Any) -> Any: ...
    def get_perp_markets(self) -> List[Any]: ...


class PositionDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class PerpMarket:
    market_index: int
    symbol: str
    oracle_price: float
    mark_price: float
    funding_rate: float
    funding_rate_apy: float
    open_interest: float
    volume_24h: float


@dataclass
class SpotMarket:
    market_index: int
    symbol: str
    oracle_price: float
    deposit_rate: float
    borrow_rate: float
    total_deposits: float
    total_borrows: float


@dataclass
class UserPosition:
    market_index: int
    market_type: str
    base_asset_amount: float
    quote_asset_amount: float
    entry_price: float
    unrealized_pnl: float
    direction: PositionDirection


@dataclass
class DriftOrder:
    order_id: int
    market_index: int
    market_type: str
    direction: PositionDirection
    base_asset_amount: float
    price: float
    status: str


def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


class DriftClientWrapper:
    def __init__(self, solana_client=None):
        self.solana_client = solana_client
        self.drift_client: Optional[DriftClientLike] = None
        self.user: Optional[DriftUserLike] = None
        self.is_subscribed = False
        self._perp_markets_cache: Dict[int, PerpMarket] = {}
        self._spot_markets_cache: Dict[int, SpotMarket] = {}
        self._last_update = 0

    async def initialize(self) -> bool:
        if Config.is_simulation():
            logger.info("[SIMULATION] Drift client initialized in simulation mode")
            self._load_mock_markets()
            return True

        if not DRIFTPY_AVAILABLE:
            logger.warning("driftpy is not installed; falling back to mock markets.")
            self._load_mock_markets()
            return False

        try:
            from solana.rpc.async_api import AsyncClient

            # These are guaranteed non-None when DRIFTPY_AVAILABLE=True, but pyright can't see that.
            DriftClientCtor = cast(Any, DriftPyClient)
            AccountSubCtor = cast(Any, AccountSubscriptionConfig)
            load_kp = cast(Any, load_keypair)

            connection = AsyncClient(Config.SOLANA_RPC_URL)

            wallet = None
            if getattr(Config, "SOLANA_KEYPAIR_PATH", None):
                wallet = load_kp(Config.SOLANA_KEYPAIR_PATH)
            else:
                logger.warning("No keypair path configured for Drift client")

            env = "mainnet" if str(getattr(Config, "DRIFT_ENV", "devnet")).lower() == "mainnet" else "devnet"

            client = DriftClientCtor(
                connection,
                wallet,
                env,
                account_subscription=AccountSubCtor("cached"),
            )

            # Narrow to protocol type for pyright, but keep runtime object as driftpy client.
            self.drift_client = cast(DriftClientLike, client)

            await self.drift_client.subscribe()
            self.is_subscribed = True
            logger.info("Drift client subscribed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Drift client: {e}")
            self._load_mock_markets()
            return False

    def _load_mock_markets(self) -> None:
        mock_perps = [
            {"index": 0, "symbol": "SOL-PERP", "price": 100.0, "funding": 0.0001},
            {"index": 1, "symbol": "BTC-PERP", "price": 43000.0, "funding": 0.00005},
            {"index": 2, "symbol": "ETH-PERP", "price": 2200.0, "funding": 0.00008},
            {"index": 3, "symbol": "APT-PERP", "price": 8.5, "funding": 0.00015},
            {"index": 4, "symbol": "ARB-PERP", "price": 1.2, "funding": -0.0001},
        ]

        for m in mock_perps:
            funding_apy = _to_float(m.get("funding")) * 24 * 365 * 100
            idx = _to_int(m.get("index"))
            px = _to_float(m.get("price"))
            self._perp_markets_cache[idx] = PerpMarket(
                market_index=idx,
                symbol=str(m.get("symbol") or f"MARKET-{idx}"),
                oracle_price=px,
                mark_price=px * 1.001,
                funding_rate=_to_float(m.get("funding")),
                funding_rate_apy=funding_apy,
                open_interest=1_000_000.0,
                volume_24h=5_000_000.0,
            )

        mock_spots = [
            {"index": 0, "symbol": "USDC", "price": 1.0, "deposit": 0.05, "borrow": 0.08},
            {"index": 1, "symbol": "SOL", "price": 100.0, "deposit": 0.03, "borrow": 0.06},
        ]

        for m in mock_spots:
            idx = _to_int(m.get("index"))
            self._spot_markets_cache[idx] = SpotMarket(
                market_index=idx,
                symbol=str(m.get("symbol") or f"SPOT-{idx}"),
                oracle_price=_to_float(m.get("price")),
                deposit_rate=_to_float(m.get("deposit")),
                borrow_rate=_to_float(m.get("borrow")),
                total_deposits=10_000_000.0,
                total_borrows=5_000_000.0,
            )

        logger.info("Loaded mock market data for simulation")

    # ---- Market getters ----

    def get_perp_markets(self) -> List[PerpMarket]:
        return list(self._perp_markets_cache.values())

    def get_perp_market(self, market_index: int) -> Optional[PerpMarket]:
        return self._perp_markets_cache.get(market_index)

    def get_spot_markets(self) -> List[SpotMarket]:
        return list(self._spot_markets_cache.values())

    def get_spot_market(self, market_index: int) -> Optional[SpotMarket]:
        return self._spot_markets_cache.get(market_index)

    def get_funding_rates(self) -> Dict[int, float]:
        return {market.market_index: market.funding_rate for market in self._perp_markets_cache.values()}

    def get_funding_apys(self) -> Dict[str, float]:
        return {market.symbol: market.funding_rate_apy for market in self._perp_markets_cache.values()}

    def get_best_funding_markets(self, top_n: int = 3, min_apy: float = 0.0) -> List[PerpMarket]:
        markets = [m for m in self._perp_markets_cache.values() if abs(m.funding_rate_apy) >= min_apy]
        return sorted(markets, key=lambda m: abs(m.funding_rate_apy), reverse=True)[:top_n]

    # ---- User / positions ----

    async def get_user_positions(self) -> List[UserPosition]:
        if Config.is_simulation():
            return []

        try:
            if self.drift_client is None:
                return []

            user = self.drift_client.get_user()
            if user is None:
                return []

            positions: List[UserPosition] = []

            # driftpy user objects expose get_perp_positions(); treat as Any to avoid tight coupling.
            for perp_position in user.get_perp_positions():
                base_amt = _to_float(getattr(perp_position, "base_asset_amount", 0.0))
                if base_amt != 0:
                    quote_amt = _to_float(getattr(perp_position, "quote_asset_amount", 0.0))
                    quote_entry = _to_float(getattr(perp_position, "quote_entry_amount", 0.0))
                    entry_price = abs(quote_entry / base_amt) if base_amt != 0 else 0.0

                    positions.append(
                        UserPosition(
                            market_index=_to_int(getattr(perp_position, "market_index", 0)),
                            market_type="perp",
                            base_asset_amount=base_amt / 1e9,
                            quote_asset_amount=quote_amt / 1e6,
                            entry_price=entry_price,
                            unrealized_pnl=0.0,
                            direction=PositionDirection.LONG if base_amt > 0 else PositionDirection.SHORT,
                        )
                    )

            return positions

        except Exception as e:
            logger.error(f"Failed to get user positions: {e}")
            return []

    # ---- Trading ----

    async def open_perp_position(
        self,
        market_index: int,
        size: float,
        direction: PositionDirection,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        if Config.is_simulation() or self.drift_client is None or not DRIFTPY_AVAILABLE:
            market = self.get_perp_market(market_index)
            symbol = market.symbol if market else f"MARKET-{market_index}"
            logger.info(f"[SIMULATION] Would open {direction.value} position: {size} on {symbol}")
            return {
                "success": True,
                "simulated": True,
                "market_index": market_index,
                "size": size,
                "direction": direction.value,
            }

        try:
            # These are only usable when driftpy is installed.
            DriftDir = cast(Any, DriftDirection)
            OP = cast(Any, OrderParams)
            OT = cast(Any, OrderType)
            MT = cast(Any, MarketType)

            drift_direction = DriftDir.Long() if direction == PositionDirection.LONG else DriftDir.Short()
            base_asset_amount = _to_int(abs(size) * 1e9)

            order_params = OP(
                order_type=OT.Market(),
                market_type=MT.Perp(),
                direction=drift_direction,
                base_asset_amount=base_asset_amount,
                market_index=market_index,
                reduce_only=reduce_only,
            )

            # FIX: pyright error ("place_perp_order" is not a known member of "None")
            # by guarding drift_client is not None above and typing drift_client as DriftClientLike.
            tx_sig = await self.drift_client.place_perp_order(order_params)

            return {
                "success": True,
                "signature": str(tx_sig),
                "market_index": market_index,
                "size": size,
                "direction": direction.value,
            }

        except Exception as e:
            logger.error(f"Failed to open perp position: {e}")
            return {"success": False, "error": str(e)}

    async def close_perp_position(self, market_index: int) -> Dict[str, Any]:
        if Config.is_simulation() or self.drift_client is None:
            logger.info(f"[SIMULATION] Would close position on market {market_index}")
            return {"success": True, "simulated": True, "market_index": market_index}

        try:
            positions = await self.get_user_positions()
            position = next((p for p in positions if p.market_index == market_index and p.market_type == "perp"), None)

            if not position:
                return {"success": True, "message": "No position to close"}

            opposite_direction = PositionDirection.SHORT if position.direction == PositionDirection.LONG else PositionDirection.LONG

            return await self.open_perp_position(
                market_index=market_index,
                size=abs(position.base_asset_amount),
                direction=opposite_direction,
                reduce_only=True,
            )

        except Exception as e:
            logger.error(f"Failed to close perp position: {e}")
            return {"success": False, "error": str(e)}

    # ---- Account metrics ----

    def get_account_value(self) -> float:
        if Config.is_simulation():
            return 10000.0

        try:
            if self.drift_client is None:
                return 0.0
            user = self.drift_client.get_user()
            if user is None:
                return 0.0
            return _to_float(user.get_total_collateral(), default=0.0) / 1e6
        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return 0.0

    def get_funding_term_structure(self) -> Dict[str, Any]:
        term_structure: Dict[str, Any] = {}

        if not self._perp_markets_cache:
            return {"term_structure": {}, "sorted_by_apy": [], "best_long": None, "best_short": None}

        for market in self._perp_markets_cache.values():
            try:
                symbol = market.symbol
                current_rate = _to_float(market.funding_rate, default=0.0)

                hourly_rate = current_rate
                daily_rate = hourly_rate * 24
                weekly_rate = daily_rate * 7
                monthly_rate = daily_rate * 30
                apy = daily_rate * 365 * 100

                term_structure[symbol] = {
                    "symbol": symbol,
                    "current_rate": current_rate,
                    "hourly_rate": round(hourly_rate * 10000, 4),
                    "daily_rate": round(daily_rate * 10000, 4),
                    "weekly_rate": round(weekly_rate * 10000, 4),
                    "monthly_rate": round(monthly_rate * 10000, 4),
                    "apy_pct": round(apy, 2),
                    "direction": "long_pays" if current_rate > 0 else "short_pays" if current_rate < 0 else "neutral",
                }
            except Exception as e:
                logger.debug(f"Error processing market {getattr(market, 'symbol', 'unknown')}: {e}")
                continue

        sorted_by_apy = sorted(term_structure.values(), key=lambda x: abs(_to_float(x.get("apy_pct"), 0.0)), reverse=True)

        return {
            "term_structure": term_structure,
            "sorted_by_apy": sorted_by_apy,
            "best_long": next((m for m in sorted_by_apy if _to_float(m.get("current_rate"), 0.0) < 0), None),
            "best_short": next((m for m in sorted_by_apy if _to_float(m.get("current_rate"), 0.0) > 0), None),
        }

    def get_margin_heatmap(self, positions: Optional[List[UserPosition]] = None) -> Dict[str, Any]:
        if positions is None:
            positions = []

        if Config.is_simulation() and not positions:
            import time as _time
            import hashlib

            hash_input = f"margin_{int(_time.time()) // 60}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)

            available_markets = list(self._perp_markets_cache.keys())[:3] if self._perp_markets_cache else [0, 1, 2]
            market_symbols: List[str] = []
            for idx in available_markets:
                market_symbols.append(self._perp_markets_cache[idx].symbol if idx in self._perp_markets_cache else f"MARKET-{idx}")

            mock_positions = [
                {"symbol": market_symbols[0] if len(market_symbols) > 0 else "SOL-PERP", "size": 100.0, "leverage": 3.0, "margin_used": 3333.0, "margin_ratio": 0.33},
                {"symbol": market_symbols[1] if len(market_symbols) > 1 else "BTC-PERP", "size": 0.5, "leverage": 5.0, "margin_used": 4300.0, "margin_ratio": 0.43},
                {"symbol": market_symbols[2] if len(market_symbols) > 2 else "ETH-PERP", "size": 2.0, "leverage": 2.0, "margin_used": 2200.0, "margin_ratio": 0.22},
            ]

            mock_positions[0]["margin_ratio"] += (hash_val % 10) / 100
            mock_positions[1]["margin_ratio"] += (hash_val % 15) / 100

            total_used = sum(_to_float(p.get("margin_used"), 0.0) for p in mock_positions)
            return {
                "positions": mock_positions,
                "total_margin_used": total_used,
                "account_value": 10000.0,
                "margin_utilization_pct": (total_used / 10000.0) * 100.0,
                "highest_leverage_position": max(mock_positions, key=lambda x: _to_float(x.get("leverage"), 0.0)) if mock_positions else None,
                "risk_zones": {
                    "safe": [p for p in mock_positions if _to_float(p.get("margin_ratio"), 0.0) < 0.5],
                    "warning": [p for p in mock_positions if 0.5 <= _to_float(p.get("margin_ratio"), 0.0) < 0.8],
                    "danger": [p for p in mock_positions if _to_float(p.get("margin_ratio"), 0.0) >= 0.8],
                },
            }

        position_data: List[Dict[str, Any]] = []
        for pos in positions:
            market = self.get_perp_market(pos.market_index)
            if market:
                notional = abs(pos.base_asset_amount) * _to_float(market.mark_price, 0.0)
                margin_used = notional / 5 if notional > 0 else 0.0
                margin_ratio = (margin_used / max(notional, 1.0)) if notional > 0 else 0.0
                leverage = (notional / margin_used) if margin_used > 0 else 0.0

                position_data.append(
                    {
                        "symbol": market.symbol,
                        "size": pos.base_asset_amount,
                        "notional": notional,
                        "leverage": leverage,
                        "margin_used": margin_used,
                        "margin_ratio": margin_ratio,
                    }
                )

        total_margin = sum(p["margin_used"] for p in position_data) if position_data else 0.0
        account_value = self.get_account_value()

        return {
            "positions": position_data,
            "total_margin_used": total_margin,
            "account_value": account_value,
            "margin_utilization_pct": (total_margin / account_value * 100.0) if account_value > 0 else 0.0,
            "highest_leverage_position": max(position_data, key=lambda x: _to_float(x.get("leverage"), 0.0)) if position_data else None,
            "risk_zones": {
                "safe": [p for p in position_data if _to_float(p.get("margin_ratio"), 0.0) < 0.5],
                "warning": [p for p in position_data if 0.5 <= _to_float(p.get("margin_ratio"), 0.0) < 0.8],
                "danger": [p for p in position_data if _to_float(p.get("margin_ratio"), 0.0) >= 0.8],
            },
        }

    def get_liquidation_ladder(self) -> Dict[str, Any]:
        ladder: Dict[str, Any] = {}

        for market in self._perp_markets_cache.values():
            symbol = market.symbol
            mark_price = _to_float(market.mark_price, 0.0)

            liq_levels = []
            for pct in [-50, -30, -20, -10, -5, 5, 10, 20, 30, 50]:
                price_level = mark_price * (1 + pct / 100)
                liq_levels.append({"price": round(price_level, 4), "pct_from_mark": pct, "direction": "below" if pct < 0 else "above"})

            ladder[symbol] = {"symbol": symbol, "mark_price": mark_price, "liquidation_levels": liq_levels}

        import time as _time

        return {"markets": ladder, "timestamp": _time.time()}

    async def close(self) -> None:
        if self.drift_client and self.is_subscribed:
            try:
                await self.drift_client.unsubscribe()
                self.is_subscribed = False
                logger.info("Drift client unsubscribed")
            except Exception as e:
                logger.error(f"Error closing Drift client: {e}")
