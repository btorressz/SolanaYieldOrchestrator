"""
Carry Optimizer Strategy
Combines funding rate capture with basis convergence to maximize "carry" returns.
Uses time-series analysis to identify markets with best expected carry.

Features:
- Historical carry & basis analysis for more robust carry estimation
- Dynamic rebalancing based on carry decay from peak
- Integration with external volatility data for risk-aware sizing
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config
from utils.logging_utils import get_logger
from utils.risk_limits import check_position_limits

logger = get_logger(__name__)


@dataclass
class CarryMetrics:
    market: str
    funding_rate_1h: float
    funding_apy: float
    basis_bps: float
    expected_basis_convergence_daily: float
    total_carry_apy: float
    volatility_adj_carry: float
    confidence_score: float
    volatility_24h: float
    funding_apy_smoothed: float
    basis_trend_bps_per_day: float


class CarryOptimizer:
    def __init__(
        self,
        drift_client,
        data_fetcher,
        analytics,
        min_carry_apy: float = None,
        max_position_pct: float = None,
        lookback_hours: int = 24,
        rebalance_threshold_bps: float = 50.0,
        target_vol: float = None,
    ):
        """
        Args:
            drift_client: client used for position limits / sizing
            data_fetcher: helper used to fetch historical funding / basis / vol (optional)
            analytics: helper used to compute stats (optional)
            min_carry_apy: minimum total carry APY (in %) to enter a trade
            max_position_pct: max % of capital to allocate per market
            lookback_hours: historical window for carry / basis stats
            rebalance_threshold_bps: carry decay from peak (in bps of APY) that triggers rebalance
            target_vol: target annualized volatility for sizing; falls back to config or 20%
        """
        self.drift_client = drift_client
        self.data_fetcher = data_fetcher
        self.analytics = analytics
        self.min_carry_apy = min_carry_apy or Config.FUNDING_MIN_APY
        self.max_position_pct = max_position_pct or 0.25
        self.lookback_hours = lookback_hours
        self.rebalance_threshold_bps = rebalance_threshold_bps
        self.target_vol = target_vol or getattr(Config, "TARGET_VOL", 0.20)

        self.strategy_name = "carry_optimizer"
        self.is_active = True

        # state
        self.positions: Dict[str, Dict] = {}
        self.carry_history: List[Dict] = []
        self.market_carry_history: Dict[str, List[CarryMetrics]] = {}
        self.last_metrics: Dict[str, CarryMetrics] = {}

        self.last_rebalance = 0
        self.pnl = 0.0

        logger.info(
            f"[{self.strategy_name}] Initialized with "
            f"min_carry_apy={self.min_carry_apy}%, "
            f"max_position_pct={self.max_position_pct*100:.1f}%, "
            f"target_vol={self.target_vol*100:.1f}%, "
            f"rebalance_threshold={rebalance_threshold_bps} bps"
        )

    # -------------------------------------------------------------------------
    # Helpers: external / historical data
    # -------------------------------------------------------------------------
    def _fetch_historical_funding(self, market: str) -> Optional[np.ndarray]:
        """
        Try to fetch historical funding rates for the market over lookback_hours.
        Returns an array of hourly funding rates (decimal per hour), or None.
        """
        try:
            # Preferred: data embedded in the snapshot / caller
            if market in self.market_carry_history:
                # extract recent funding_apy and convert back to per-hour rate approx
                recent = self.market_carry_history[market][-self.lookback_hours :]
                if recent:
                    # funding_apy ~ rate_1h * 24 * 365 * 100 => rate_1h ~ apy / (24*365*100)
                    rates = [
                        m.funding_apy / (24 * 365 * 100)
                        for m in recent
                        if m.funding_apy is not None
                    ]
                    if len(rates) >= 4:
                        return np.array(rates, dtype=float)

            # Fallback: data_fetcher (if it knows how to do this)
            if hasattr(self.data_fetcher, "get_funding_history"):
                series = self.data_fetcher.get_funding_history(
                    market, hours=self.lookback_hours
                )
                if series:
                    return np.array(series, dtype=float)
        except Exception as e:
            logger.debug(
                f"[{self.strategy_name}] Error fetching funding history for {market}: {e}"
            )
        return None

    def _fetch_historical_basis(self, market: str, spot_price: float, perp_price: float) -> Optional[np.ndarray]:
        """
        Try to fetch historical basis (bps) for the market over lookback_hours.
        Returns an array of basis in bps, or None.
        """
        try:
            # From data_fetcher if available
            if hasattr(self.data_fetcher, "get_basis_history"):
                series = self.data_fetcher.get_basis_history(
                    market, hours=self.lookback_hours
                )
                if series:
                    return np.array(series, dtype=float)

            # From analytics if price history is aggregated elsewhere
            if hasattr(self.analytics, "get_basis_series"):
                series = self.analytics.get_basis_series(
                    market, hours=self.lookback_hours
                )
                if series is not None:
                    return np.array(series, dtype=float)
        except Exception as e:
            logger.debug(
                f"[{self.strategy_name}] Error fetching basis history for {market}: {e}"
            )

        # As a last resort, use a constant basis series around the current value
        if spot_price > 0:
            basis_bps = ((perp_price - spot_price) / spot_price) * 10000
            return np.full(self.lookback_hours, basis_bps, dtype=float)

        return None

    def _fetch_volatility_24h(self, market: str, price_data: Dict) -> float:
        """
        Integrate with external volatility data when available:
        1. Use explicit volatility_24h from price_data if present
        2. Ask data_fetcher / analytics for realized or implied vol
        3. Fall back to a conservative default (2%)
        """
        # 1) Provided by caller
        if "volatility_24h" in price_data and price_data["volatility_24h"] is not None:
            return float(price_data["volatility_24h"])

        # 2) data_fetcher
        try:
            if hasattr(self.data_fetcher, "get_volatility_24h"):
                v = self.data_fetcher.get_volatility_24h(market)
                if v is not None:
                    return float(v)
        except Exception as e:
            logger.debug(
                f"[{self.strategy_name}] Error fetching volatility_24h via data_fetcher "
                f"for {market}: {e}"
            )

        # 3) analytics
        try:
            if hasattr(self.analytics, "get_volatility_24h"):
                v = self.analytics.get_volatility_24h(market)
                if v is not None:
                    return float(v)
        except Exception as e:
            logger.debug(
                f"[{self.strategy_name}] Error fetching volatility_24h via analytics "
                f"for {market}: {e}"
            )

        # default fallback
        return 0.02  # 2% annualized placeholder

    # -------------------------------------------------------------------------
    # Carry calculation
    # -------------------------------------------------------------------------
    def calculate_carry_metrics(
        self,
        market: str,
        funding_data: Dict,
        price_data: Dict,
    ) -> Optional[CarryMetrics]:
        """
        More sophisticated carry calculation using historical funding and basis data.

        - Uses historical funding to smooth APY and boost confidence if stable
        - Uses historical basis to estimate expected convergence and trend
        - Integrates external volatility for volatility-adjusted carry
        """
        try:
            # Instantaneous funding
            funding_rate_inst = float(funding_data.get("rate_1h", 0.0))
            funding_apy_inst = funding_rate_inst * 24 * 365 * 100  # decimal -> % APY

            # Historical funding smoothing
            funding_hist = self._fetch_historical_funding(market)
            if funding_hist is not None and len(funding_hist) >= 4:
                funding_rate_mean = float(funding_hist.mean())
                funding_apy_smoothed = funding_rate_mean * 24 * 365 * 100
                funding_apy = 0.5 * funding_apy_inst + 0.5 * funding_apy_smoothed
            else:
                funding_apy_smoothed = funding_apy_inst
                funding_apy = funding_apy_inst

            # Spot / perp / basis
            spot_price = float(price_data.get("spot", 0.0))
            perp_price = float(price_data.get("perp", spot_price))

            if spot_price <= 0:
                return None

            basis_bps_current = ((perp_price - spot_price) / spot_price) * 10000.0

            # Historical basis for convergence + trend
            basis_hist = self._fetch_historical_basis(
                market, spot_price=spot_price, perp_price=perp_price
            )
            basis_trend_bps_per_day = 0.0
            expected_convergence_daily_bps = 0.0

            if basis_hist is not None and len(basis_hist) >= 5:
                # Simple trend (bps per hour) via regression
                x = np.arange(len(basis_hist))
                slope, intercept = np.polyfit(x, basis_hist, 1)
                # Convert to per-day
                basis_trend_bps_per_day = float(slope * 24.0)

                # Deviation from mean
                basis_mean = float(basis_hist.mean())
                deviation = basis_bps_current - basis_mean

                # Assume partial mean reversion towards mean each day
                # If basis is above mean, expect convergence downwards (and vice versa)
                # convergence sign is opposite of deviation; damped by trend
                base_convergence = -0.5 * deviation  # half-distance per day
                # If trend is pushing basis away from mean, damp convergence
                if np.sign(slope) == np.sign(deviation):
                    base_convergence *= 0.5

                expected_convergence_daily_bps = base_convergence
            else:
                # Fallback: simple fraction of basis
                expected_convergence_daily_bps = basis_bps_current * 0.1

            # Convert expected basis convergence to APY (%)
            convergence_apy = (expected_convergence_daily_bps / 10000.0) * 365.0 * 100.0

            # Total carry = funding + basis carry
            total_carry_apy = funding_apy + convergence_apy

            # External volatility
            volatility_24h = self._fetch_volatility_24h(market, price_data)
            if volatility_24h > 0:
                volatility_adj_carry = total_carry_apy / (volatility_24h * 100.0)
            else:
                volatility_adj_carry = total_carry_apy

            # Confidence score: depends on funding stability and sample sizes
            conf = 0.5
            if funding_hist is not None and len(funding_hist) >= 8:
                # lower std => higher confidence
                funding_std = float(funding_hist.std())
                conf_funding = max(0.0, min(1.0, 1.0 - funding_std * 10.0))
                conf += 0.25 * conf_funding

            if basis_hist is not None and len(basis_hist) >= 8:
                # if trend is modest compared to current basis, higher confidence
                trend_ratio = abs(basis_trend_bps_per_day) / (abs(basis_bps_current) + 1e-6)
                conf_basis = max(0.0, min(1.0, 1.0 - trend_ratio))
                conf += 0.25 * conf_basis

            # Combine with magnitude of carry
            magnitude_factor = min(1.0, abs(total_carry_apy) / 50.0)  # saturate at 50% APY
            confidence = max(0.0, min(1.0, conf * magnitude_factor))

            metrics = CarryMetrics(
                market=market,
                funding_rate_1h=funding_rate_inst,
                funding_apy=funding_apy,
                basis_bps=basis_bps_current,
                expected_basis_convergence_daily=expected_convergence_daily_bps,
                total_carry_apy=total_carry_apy,
                volatility_adj_carry=volatility_adj_carry,
                confidence_score=confidence,
                volatility_24h=volatility_24h,
                funding_apy_smoothed=funding_apy_smoothed,
                basis_trend_bps_per_day=basis_trend_bps_per_day,
            )

            # Track per-market carry history for later stats
            self.market_carry_history.setdefault(market, []).append(metrics)
            self.last_metrics[market] = metrics

            return metrics

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Error calculating carry for {market}: {e}")
            return None

    def rank_markets_by_carry(self, market_data: Dict[str, Dict]) -> List[CarryMetrics]:
        """
        Rank markets by volatility-adjusted carry, with a minimum carry filter.
        """
        carry_metrics: List[CarryMetrics] = []

        for market, data in market_data.items():
            funding_data = data.get("funding", {})
            price_data = data.get("prices", {})

            metrics = self.calculate_carry_metrics(market, funding_data, price_data)
            if metrics and abs(metrics.total_carry_apy) >= self.min_carry_apy:
                carry_metrics.append(metrics)

        # Primary key: volatility-adjusted carry
        carry_metrics.sort(key=lambda x: x.volatility_adj_carry, reverse=True)
        return carry_metrics

    # -------------------------------------------------------------------------
    # Sizing & rebalancing logic
    # -------------------------------------------------------------------------
    def _compute_size_pct(self, metrics: CarryMetrics) -> float:
        """
        Volatility-aware sizing:
        - For vol <= target, allocate full max_position_pct
        - For vol > target, scale size down proportionally
        """
        vol = max(metrics.volatility_24h, 1e-6)

        if vol <= self.target_vol:
            size_pct = self.max_position_pct
        else:
            size_pct = self.max_position_pct * (self.target_vol / vol)
            size_pct = max(0.05 * self.max_position_pct, size_pct)  # floor at 5% of max

        return min(self.max_position_pct, size_pct)

    def _should_rebalance_position(
        self,
        market: str,
        metrics: Optional[CarryMetrics],
        position: Dict,
    ) -> Dict[str, Any]:
        """
        Decide whether to close or resize an existing position based on:
        - Carry decay from peak (dynamic rebalancing)
        - Falling below minimum carry threshold
        """
        if metrics is None:
            # No metrics => close defensively
            return {
                "action": "close",
                "reason": "metrics_unavailable",
            }

        current_carry = metrics.total_carry_apy
        entry_carry = position.get("entry_carry_apy", current_carry)
        max_carry = position.get("max_carry_apy", entry_carry)

        # Update max carry in position
        if current_carry > max_carry:
            position["max_carry_apy"] = current_carry
            max_carry = current_carry

        # Absolute threshold: if carry < 0.5 * min_carry_apy -> close
        if abs(current_carry) < self.min_carry_apy * 0.5:
            return {
                "action": "close",
                "reason": "carry_below_threshold",
            }

        # Carry decay from peak (in APY points)
        decay_points = max_carry - current_carry
        # Convert threshold from bps to percentage points
        decay_threshold_points = self.rebalance_threshold_bps / 100.0

        if decay_points >= decay_threshold_points and abs(current_carry) >= self.min_carry_apy:
            # Still decent carry but decayed from peak: reduce size
            return {
                "action": "reduce",
                "reason": "carry_decay",
                "decay_points": decay_points,
            }

        # No rebalance needed
        return {
            "action": "hold",
            "reason": "carry_stable",
        }

    # -------------------------------------------------------------------------
    # Main strategy loop
    # -------------------------------------------------------------------------
    def generate_signals(self, market_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_active:
            return []

        actions: List[Dict[str, Any]] = []

        try:
            funding_rates = market_snapshot.get("funding_rates", {})
            prices = market_snapshot.get("prices", {})
            capital = market_snapshot.get("capital", None)  # optional

            # Build normalized market_data
            market_data: Dict[str, Dict[str, Any]] = {}
            for market, funding in funding_rates.items():
                symbol = market.replace("-PERP", "").upper()
                spot_price = prices.get(symbol, 0.0)
                perp_price = funding.get("mark_price", spot_price)

                market_data[market] = {
                    "funding": {
                        "rate_1h": funding.get("rate", 0.0),
                        # If history is attached, pass it through
                        "history": funding.get("history"),
                    },
                    "prices": {
                        "spot": spot_price,
                        "perp": perp_price,
                        "volatility_24h": funding.get("volatility_24h"),
                    },
                }

            ranked_markets = self.rank_markets_by_carry(market_data)

            # -----------------------------------------------------------------
            # Open new positions in top markets
            # -----------------------------------------------------------------
            for metrics in ranked_markets[:3]:
                if metrics.market in self.positions:
                    continue

                direction = "short" if metrics.funding_rate_1h > 0 else "long"
                size_pct = self._compute_size_pct(metrics)

                # Risk limits / notional check
                if not check_position_limits(self.drift_client, metrics.market, size_pct):
                    logger.info(
                        f"[{self.strategy_name}] Skipping {metrics.market}: "
                        f"position limits exceeded for size_pct={size_pct:.4f}"
                    )
                    continue

                actions.append(
                    {
                        "action_type": "open_perp",
                        "params": {
                            "market": metrics.market,
                            "direction": direction,
                            "size_pct": size_pct,
                            "strategy": self.strategy_name,
                            "carry_apy": metrics.total_carry_apy,
                            "vol_adj_carry": metrics.volatility_adj_carry,
                            "volatility_24h": metrics.volatility_24h,
                        },
                        "priority": "balanced",
                    }
                )

                logger.info(
                    f"[{self.strategy_name}] Signal: {direction} {metrics.market}, "
                    f"carry={metrics.total_carry_apy:.2f}% APY "
                    f"(vol_adj={metrics.volatility_adj_carry:.2f}, vol={metrics.volatility_24h*100:.1f}%)"
                )

            # -----------------------------------------------------------------
            # Rebalance / close existing positions based on carry decay
            # -----------------------------------------------------------------
            for market, pos in list(self.positions.items()):
                metrics = next((m for m in ranked_markets if m.market == market), None)
                decision = self._should_rebalance_position(market, metrics, pos)

                if decision["action"] == "close":
                    actions.append(
                        {
                            "action_type": "close_perp",
                            "params": {
                                "market": market,
                                "strategy": self.strategy_name,
                                "reason": decision["reason"],
                            },
                            "priority": "balanced",
                        }
                    )
                    logger.info(
                        f"[{self.strategy_name}] Closing {market}: reason={decision['reason']}"
                    )

                elif decision["action"] == "reduce":
                    current_size = pos.get("size", 0.0)
                    if current_size <= 0:
                        continue

                    # Reduce to 50% of current size (simple heuristic)
                    target_size = current_size * 0.5
                    reduction = current_size - target_size

                    actions.append(
                        {
                            "action_type": "reduce_position",
                            "params": {
                                "market": market,
                                "current_size": current_size,
                                "target_size": target_size,
                                "reduction": reduction,
                                "reason": decision["reason"],
                                "strategy": self.strategy_name,
                            },
                            "priority": "balanced",
                        }
                    )

                    logger.info(
                        f"[{self.strategy_name}] Reducing {market}: "
                        f"carry decay={decision.get('decay_points', 0.0):.2f} pts, "
                        f"size={current_size:.4f}->{target_size:.4f}"
                    )

            # -----------------------------------------------------------------
            # Log latest rankings for diagnostics
            # -----------------------------------------------------------------
            self.carry_history.append(
                {
                    "timestamp": time.time(),
                    "ranked_markets": [
                        {
                            "market": m.market,
                            "carry_apy": m.total_carry_apy,
                            "vol_adj_carry": m.volatility_adj_carry,
                        }
                        for m in ranked_markets[:5]
                    ],
                }
            )

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Error generating signals: {e}")

        return actions

    # -------------------------------------------------------------------------
    # Simulation / status
    # -------------------------------------------------------------------------
    def apply_simulated_trade(self, action: Dict, result: Dict):
        market = action["params"].get("market", "")
        action_type = action.get("action_type")

        if action_type == "open_perp" and result.get("success"):
            carry_apy = action["params"].get("carry_apy", 0.0)
            now = time.time()
            self.positions[market] = {
                "direction": action["params"].get("direction"),
                "size": result.get("size", 0.0),
                "entry_price": result.get("price", 0.0),
                "entry_carry_apy": carry_apy,
                "max_carry_apy": carry_apy,
                "opened_at": now,
            }

        elif action_type in ("close_perp", "reduce_position") and result.get("success"):
            if market in self.positions:
                pnl = float(result.get("pnl", 0.0))
                self.pnl += pnl

                if action_type == "close_perp":
                    del self.positions[market]
                else:
                    # For reduce_position, adjust stored size
                    reduction = float(action["params"].get("reduction", 0.0))
                    pos = self.positions[market]
                    if reduction > 0 and reduction < pos.get("size", 0.0):
                        pos["size"] = pos["size"] - reduction
                        self.positions[market] = pos
                    else:
                        # if reduction >= size, treat as full close
                        del self.positions[market]

    def get_status(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "is_active": self.is_active,
            "positions": self.positions,
            "total_pnl": self.pnl,
            "min_carry_apy": self.min_carry_apy,
            "rebalance_threshold_bps": self.rebalance_threshold_bps,
            "target_vol": self.target_vol,
            "recent_rankings": self.carry_history[-5:] if self.carry_history else [],
        }
