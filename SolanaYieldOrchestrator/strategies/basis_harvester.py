import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import Config
from strategies import Strategy, Action
from data.analytics import Analytics, BasisAnalysis
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _coerce_float(x: Any, default: float) -> float:
    """Return a non-None float for pyright + runtime safety."""
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _coerce_int(x: Any, default: int) -> int:
    """Return a non-None int for pyright + runtime safety."""
    if x is None:
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


@dataclass
class BasisPosition:
    market: str
    spot_size: float
    perp_size: float
    entry_spot_price: float
    entry_perp_price: float
    entry_basis_bps: float
    entry_time: float
    current_pnl: float = 0.0


class BasisHarvester(Strategy):
    def __init__(
        self,
        target_alloc: Optional[float] = None,
        entry_threshold_bps: Optional[int] = None,
        exit_threshold_bps: Optional[int] = None,
    ):
        # FIX (Ln 26-27): never allow None to flow into float/int fields
        self._target_allocation: float = _coerce_float(
            target_alloc, _coerce_float(getattr(Config, "ALLOCATION_BASIS_HARVESTER", 0.0), 0.0)
        )
        self.entry_threshold_bps: int = _coerce_int(
            entry_threshold_bps, _coerce_int(getattr(Config, "BASIS_ENTRY_THRESHOLD_BPS", 0), 0)
        )
        self.exit_threshold_bps: int = _coerce_int(
            exit_threshold_bps, _coerce_int(getattr(Config, "BASIS_EXIT_THRESHOLD_BPS", 0), 0)
        )

        self.analytics = Analytics()
        self.positions: Dict[str, BasisPosition] = {}

        self._current_snapshot = None
        self._last_update: float = 0.0
        self._total_pnl: float = 0.0
        self._trade_count: int = 0

    def name(self) -> str:
        return "basis_harvester"

    def target_allocation(self) -> float:
        return float(self._target_allocation)

    def update_state(self, market_snapshot) -> None:
        self._current_snapshot = market_snapshot
        self._last_update = time.time()
        self._update_position_pnl()

    def _update_position_pnl(self) -> None:
        if not self._current_snapshot:
            return

        for market, position in list(self.positions.items()):
            current_spot = self._current_snapshot.spot_prices.get("SOL")
            current_perp = self._current_snapshot.perp_prices.get(market, 0)

            if current_spot and current_perp:
                spot_price = current_spot.price if hasattr(current_spot, "price") else current_spot

                # Normalize to floats to avoid odd types coming from snapshots
                spot_price_f = _coerce_float(spot_price, 0.0)
                perp_price_f = _coerce_float(current_perp, 0.0)

                if spot_price_f <= 0 or perp_price_f <= 0:
                    continue

                spot_pnl = (spot_price_f - position.entry_spot_price) * position.spot_size
                perp_pnl = (position.entry_perp_price - perp_price_f) * abs(position.perp_size)
                position.current_pnl = float(spot_pnl + perp_pnl)

    def _analyze_basis_opportunity(self, market: str = "SOL-PERP") -> Optional[BasisAnalysis]:
        if not self._current_snapshot:
            return None

        spot_data = self._current_snapshot.spot_prices.get("SOL")
        if not spot_data:
            return None

        spot_price = spot_data.price if hasattr(spot_data, "price") else spot_data
        perp_price = self._current_snapshot.perp_prices.get(market, 0)

        spot_price_f = _coerce_float(spot_price, 0.0)
        perp_price_f = _coerce_float(perp_price, 0.0)

        if spot_price_f <= 0 or perp_price_f <= 0:
            return None

        funding_data = self._current_snapshot.funding_rates.get(market)
        funding_rate = 0.0
        if funding_data:
            if hasattr(funding_data, "funding_rate"):
                funding_rate = _coerce_float(getattr(funding_data, "funding_rate", 0.0), 0.0)
            else:
                # dict-like
                try:
                    funding_rate = _coerce_float(funding_data.get("rate", 0.0), 0.0)
                except Exception:
                    funding_rate = 0.0

        return self.analytics.calculate_basis(spot_price_f, perp_price_f, funding_rate)

    def desired_actions(self, vault_state: Dict[str, Any]) -> List[Action]:
        actions: List[Action] = []

        if not self._current_snapshot:
            return actions

        available_capital = _coerce_float(vault_state.get("available_capital", 0), 0.0) * float(self._target_allocation)
        current_allocation = _coerce_float(
            (vault_state.get("strategy_allocations", {}) or {}).get(self.name(), 0),
            0.0,
        )
        _ = current_allocation  # kept for future allocation logic

        market = "SOL-PERP"
        basis = self._analyze_basis_opportunity(market)
        if not basis:
            return actions

        if market in self.positions:
            position = self.positions[market]

            should_unwind = (abs(basis.basis_bps) < self.exit_threshold_bps) or (
                basis.is_contango != (position.perp_size < 0)
            )

            if should_unwind:
                logger.info(
                    f"[{self.name()}] Unwinding position: basis={basis.basis_bps:.2f}bps, "
                    f"threshold={self.exit_threshold_bps}bps"
                )

                actions.append(
                    Action(
                        action_type="close_perp",
                        params={"market_index": 0, "market": market},
                        priority="balanced",
                    )
                )

                actions.append(
                    Action(
                        action_type="swap",
                        params={
                            "input_mint": "So11111111111111111111111111111111111111112",
                            "output_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                            "amount": int(abs(position.spot_size) * 1e9),
                        },
                        priority="balanced",
                    )
                )
        else:
            if abs(basis.basis_bps) >= self.entry_threshold_bps and available_capital > 100:
                position_size_usd = min(available_capital * 0.5, _coerce_float(getattr(Config, "MAX_POSITION_SIZE_USD", 0.0), 0.0))
                spot_size = position_size_usd / basis.spot_price if basis.spot_price > 0 else 0.0

                if spot_size > 0:
                    logger.info(
                        f"[{self.name()}] Opening position: basis={basis.basis_bps:.2f}bps, "
                        f"size=${position_size_usd:.2f}"
                    )

                    if basis.is_contango:
                        actions.append(
                            Action(
                                action_type="swap",
                                params={
                                    "input_mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                                    "output_mint": "So11111111111111111111111111111111111111112",
                                    "amount": int(position_size_usd * 1e6),
                                },
                                priority="balanced",
                            )
                        )

                        actions.append(
                            Action(
                                action_type="open_perp",
                                params={
                                    "market_index": 0,
                                    "market": market,
                                    "size": spot_size,
                                    "direction": "short",
                                },
                                priority="balanced",
                            )
                        )
                    else:
                        actions.append(
                            Action(
                                action_type="open_perp",
                                params={
                                    "market_index": 0,
                                    "market": market,
                                    "size": spot_size,
                                    "direction": "long",
                                },
                                priority="balanced",
                            )
                        )

        return actions

    def apply_simulated_trade(self, action: Action, success: bool = True) -> None:
        if not success:
            return

        market = action.params.get("market", "SOL-PERP")

        if action.action_type == "open_perp":
            size = _coerce_float(action.params.get("size", 0), 0.0)
            direction = str(action.params.get("direction", "short"))

            spot_data = self._current_snapshot.spot_prices.get("SOL") if self._current_snapshot else None
            spot_price = (
                _coerce_float(getattr(spot_data, "price", None), 100.0) if spot_data else 100.0
            )
            perp_price = (
                _coerce_float(self._current_snapshot.perp_prices.get(market, spot_price), spot_price)
                if self._current_snapshot
                else spot_price
            )

            basis = self.analytics.calculate_basis(spot_price, perp_price)

            perp_size = -size if direction == "short" else size

            self.positions[market] = BasisPosition(
                market=market,
                spot_size=size if direction == "short" else -size,
                perp_size=perp_size,
                entry_spot_price=spot_price,
                entry_perp_price=perp_price,
                entry_basis_bps=basis.basis_bps,
                entry_time=time.time(),
            )

            self._trade_count += 1
            logger.info(f"[{self.name()}] Opened simulated position: {direction} {size} @ {perp_price}")

        elif action.action_type == "close_perp":
            if market in self.positions:
                position = self.positions[market]
                self._total_pnl += position.current_pnl
                del self.positions[market]
                self._trade_count += 1
                logger.info(f"[{self.name()}] Closed simulated position: PnL={position.current_pnl:.2f}")

    def get_status(self) -> Dict[str, Any]:
        basis = self._analyze_basis_opportunity()

        position_value = sum(abs(p.spot_size * p.entry_spot_price) for p in self.positions.values())
        unrealized_pnl = sum(p.current_pnl for p in self.positions.values())

        return {
            "name": self.name(),
            "target_allocation": float(self._target_allocation),
            "entry_threshold_bps": int(self.entry_threshold_bps),
            "exit_threshold_bps": int(self.exit_threshold_bps),
            "current_basis_bps": basis.basis_bps if basis else 0.0,
            "current_basis_apy": basis.annualized_apy if basis else 0.0,
            "is_contango": bool(basis.is_contango) if basis else False,
            "active_positions": int(len(self.positions)),
            "position_value_usd": float(position_value),
            "unrealized_pnl": float(unrealized_pnl),
            "realized_pnl": float(self._total_pnl),
            "total_trades": int(self._trade_count),
            "last_update": float(self._last_update),
        }

