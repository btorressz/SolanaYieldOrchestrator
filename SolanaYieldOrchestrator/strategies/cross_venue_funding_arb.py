from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

from config import Config
from infra.perp_venue import PerpVenue, VenueRegistry, VenueName
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CrossVenuePosition:
    symbol: str
    long_venue: str
    short_venue: str
    size: float
    entry_funding_diff: float
    current_funding_diff: float
    accumulated_funding: float
    unrealized_pnl: float
    opened_at: float


class CrossVenueFundingArbStrategy:
    def __init__(
        self,
        venues: Optional[List[PerpVenue]] = None,
        symbols: Optional[List[str]] = None,
        min_funding_diff_bps: float = 5.0,
        max_position_usd: float = 10000.0,
    ):
        self.venues: List[PerpVenue] = venues or []
        self.symbols: List[str] = symbols or ["SOL", "BTC", "ETH"]
        self.min_funding_diff_bps = float(min_funding_diff_bps)
        self.max_position_usd = float(max_position_usd)

        self.positions: List[CrossVenuePosition] = []
        self.pnl_history: List[Dict[str, Any]] = []
        self.total_funding_pnl: float = 0.0
        self.total_trading_pnl: float = 0.0

    # FIX: market_data default must be a Dict[str, Any], not None
    def analyze(self, market_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        market_data = market_data or {}  # ensures Dict[str, Any] for pyright

        if len(self.venues) < 2:
            venues = VenueRegistry.get_all()
            if len(venues) < 2:
                return actions
            self.venues = venues

        for symbol in self.symbols:
            funding_comparison = self._compare_funding_rates(symbol)

            if not funding_comparison:
                continue

            diff_bps = abs(float(funding_comparison.get("diff_bps", 0.0)))

            existing_position = self._get_position(symbol)

            if existing_position:
                existing_position.current_funding_diff = float(funding_comparison["diff_bps"])

                if diff_bps < float(self.min_funding_diff_bps) / 2.0:
                    actions.append(
                        {
                            "action_type": "close_cross_venue",
                            "params": {
                                "symbol": symbol,
                                "reason": "funding_converged",
                                "current_diff_bps": diff_bps,
                                "strategy": "cross_venue_funding_arb",
                            },
                            "priority": "balanced",
                        }
                    )
            else:
                if diff_bps >= float(self.min_funding_diff_bps):
                    best_long_venue, best_short_venue = funding_comparison["suggested_venues"]

                    actions.append(
                        {
                            "action_type": "open_cross_venue",
                            "params": {
                                "symbol": symbol,
                                "long_venue": best_long_venue,
                                "short_venue": best_short_venue,
                                "funding_diff_bps": diff_bps,
                                "size_usd": float(self.max_position_usd),
                                "strategy": "cross_venue_funding_arb",
                            },
                            "priority": "balanced",
                        }
                    )

        return actions

    def _compare_funding_rates(self, symbol: str) -> Optional[Dict[str, Any]]:
        venue_rates: Dict[str, Dict[str, float]] = {}

        for venue in self.venues:
            funding = venue.get_funding(symbol)
            if funding:
                venue_rates[venue.name.value] = {
                    "rate": float(funding.get("rate", 0.0)),
                    "apy": float(funding.get("apy", 0.0)),
                }

        if len(venue_rates) < 2:
            return None

        max_rate_venue = max(venue_rates.keys(), key=lambda k: float(venue_rates[k]["rate"]))
        min_rate_venue = min(venue_rates.keys(), key=lambda k: float(venue_rates[k]["rate"]))

        diff = float(venue_rates[max_rate_venue]["rate"]) - float(venue_rates[min_rate_venue]["rate"])
        diff_bps = diff * 10000.0

        return {
            "symbol": symbol,
            "venues": venue_rates,
            "diff": diff,
            "diff_bps": diff_bps,
            "diff_apy": diff * 24.0 * 365.0 * 100.0,
            "suggested_venues": (min_rate_venue, max_rate_venue),  # long where funding is lower, short where higher
        }

    def _get_position(self, symbol: str) -> Optional[CrossVenuePosition]:
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action.get("action_type")
        params = action.get("params", {}) or {}

        if action_type == "open_cross_venue":
            return self._open_position(params)
        if action_type == "close_cross_venue":
            return self._close_position(params)

        return {"success": False, "error": f"Unknown action: {action_type}"}

    def _open_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(params["symbol"])
        long_venue = str(params["long_venue"])
        short_venue = str(params["short_venue"])
        funding_diff_bps = float(params["funding_diff_bps"])
        size_usd = float(params.get("size_usd", self.max_position_usd))

        if Config.is_simulation():
            position = CrossVenuePosition(
                symbol=symbol,
                long_venue=long_venue,
                short_venue=short_venue,
                size=size_usd,
                entry_funding_diff=funding_diff_bps,
                current_funding_diff=funding_diff_bps,
                accumulated_funding=0.0,
                unrealized_pnl=0.0,
                opened_at=time.time(),
            )
            self.positions.append(position)

            logger.info(
                f"[cross_venue_arb] Opened: {symbol} long on {long_venue}, short on {short_venue}, diff={funding_diff_bps:.1f}bps"
            )

            return {
                "success": True,
                "simulated": True,
                "symbol": symbol,
                "long_venue": long_venue,
                "short_venue": short_venue,
                "size": size_usd,
                "funding_diff_bps": funding_diff_bps,
            }

        long_venue_obj = VenueRegistry.get(VenueName(long_venue))
        short_venue_obj = VenueRegistry.get(VenueName(short_venue))

        if not long_venue_obj or not short_venue_obj:
            return {"success": False, "error": "Venue not found"}

        mark_price = float(long_venue_obj.get_mark_price(symbol) or 100.0)
        size = float(size_usd / mark_price) if mark_price > 0 else 0.0
        if size <= 0:
            return {"success": False, "error": "Invalid size computed"}

        long_result = long_venue_obj.open_position(symbol, size, "long")
        if not long_result.success:
            return {"success": False, "error": f"Long order failed: {long_result.error}"}

        short_result = short_venue_obj.open_position(symbol, size, "short")
        if not short_result.success:
            long_venue_obj.close_position(symbol)
            return {"success": False, "error": f"Short order failed: {short_result.error}"}

        position = CrossVenuePosition(
            symbol=symbol,
            long_venue=long_venue,
            short_venue=short_venue,
            size=size_usd,
            entry_funding_diff=funding_diff_bps,
            current_funding_diff=funding_diff_bps,
            accumulated_funding=0.0,
            unrealized_pnl=0.0,
            opened_at=time.time(),
        )
        self.positions.append(position)

        return {
            "success": True,
            "symbol": symbol,
            "long_venue": long_venue,
            "short_venue": short_venue,
            "size": size_usd,
        }

    def _close_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(params["symbol"])

        position = self._get_position(symbol)
        if not position:
            return {"success": False, "error": "Position not found"}

        hold_time_hours = (time.time() - float(position.opened_at)) / 3600.0
        estimated_funding_pnl = (float(position.entry_funding_diff) / 10000.0) * float(position.size) * (hold_time_hours / 8.0)

        self.total_funding_pnl += float(estimated_funding_pnl)

        self.pnl_history.append(
            {
                "timestamp": time.time(),
                "symbol": symbol,
                "long_venue": position.long_venue,
                "short_venue": position.short_venue,
                "hold_time_hours": hold_time_hours,
                "entry_diff_bps": float(position.entry_funding_diff),
                "exit_diff_bps": float(position.current_funding_diff),
                "estimated_funding_pnl": float(estimated_funding_pnl),
            }
        )

        self.positions.remove(position)

        logger.info(f"[cross_venue_arb] Closed: {symbol}, funding PnL=${estimated_funding_pnl:.2f}")

        return {
            "success": True,
            "simulated": Config.is_simulation(),
            "symbol": symbol,
            "funding_pnl": float(estimated_funding_pnl),
            "total_funding_pnl": float(self.total_funding_pnl),
        }

    def get_cross_venue_opportunities(self) -> List[Dict[str, Any]]:
        opportunities: List[Dict[str, Any]] = []

        for symbol in self.symbols:
            comparison = self._compare_funding_rates(symbol)
            if comparison and abs(float(comparison["diff_bps"])) >= float(self.min_funding_diff_bps):
                opportunities.append(
                    {
                        "symbol": symbol,
                        "venues": comparison["venues"],
                        "diff_bps": float(comparison["diff_bps"]),
                        "diff_apy": float(comparison["diff_apy"]),
                        "suggested": f"Long {comparison['suggested_venues'][0]}, Short {comparison['suggested_venues'][1]}",
                    }
                )

        return sorted(opportunities, key=lambda x: -abs(float(x["diff_bps"])))

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "cross_venue_funding_arb",
            "enabled": len(self.venues) >= 2,
            "venues": [v.name.value for v in self.venues],
            "symbols": self.symbols,
            "positions_count": len(self.positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "long_venue": p.long_venue,
                    "short_venue": p.short_venue,
                    "size": float(p.size),
                    "entry_diff_bps": float(p.entry_funding_diff),
                    "current_diff_bps": float(p.current_funding_diff),
                    "accumulated_funding": float(p.accumulated_funding),
                    "hold_time_hours": (time.time() - float(p.opened_at)) / 3600.0,
                }
                for p in self.positions
            ],
            "total_funding_pnl": float(self.total_funding_pnl),
            "recent_trades": self.pnl_history[-10:],
        }

