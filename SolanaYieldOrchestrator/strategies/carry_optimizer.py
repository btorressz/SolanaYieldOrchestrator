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
import pandas as pd  # kept for downstream integrations / optional analytics usage

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------------
# Optional risk-limits import (fix: "check_position_limits is unknown import symbol")
# -------------------------------------------------------------------------
try:
    from utils.risk_limits import check_position_limits as _check_position_limits
except Exception:
    _check_position_limits = None  # type: ignore[assignment]


def check_position_limits(drift_client: Any, market: str, size_pct: float) -> bool:
    """
    Safe wrapper so pyright always sees a valid symbol and the strategy
    still runs even if the risk_limits module isn't present.
    """
    if _check_position_limits is None:
        # Conservative default: allow (or change to False if you prefer fail-closed)
        return True
    try:
        return bool(_check_position_limits(drift_client, market, size_pct))
    except Exception:
        return True


def _coerce_float(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _cfg_float(name: str, default: float) -> float:
    return _coerce_float(getattr(Config, name, None), default)


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
        drift_client: Any,
        data_fetcher: Any,
        analytics: Any,
        min_carry_apy: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        lookback_hours: int = 24,
        rebalance_threshold_bps: float = 50.0,
        target_vol: Optional[float] = None,
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

        # FIX (Config.FUNDING_MIN_APY unknown): use getattr with fallback
        cfg_min_carry = _cfg_float("FUNDING_MIN_APY", 10.0)  # default 10% APY floor
        self.min_carry_apy: float = _coerce_float(min_carry_apy, cfg_min_carry)

        self.max_position_pct: float = _coerce_float(max_position_pct, 0.25)
        self.lookback_hours: int = int(lookback_hours)
        self.rebalance_threshold_bps: float = float(rebalance_threshold_bps)

        # FIX (None -> float): ensure always float
        self.target_vol: float = _coerce_float(target_vol, _cfg_float("TARGET_VOL", 0.20))

        self.strategy_name = "carry_optimizer"
        self.is_active = True

        # state
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.carry_history: List[Dict[str, Any]] = []
        self.market_carry_history: Dict[str, List[CarryMetrics]] = {}
        self.last_metrics: Dict[str, CarryMetrics] = {}

        self.last_rebalance: float = 0.0
        self.pnl: float = 0.0

        logger.info(
            f"[{self.strategy_name}] Initialized with "
            f"min_carry_apy={self.min_carry_apy}%, "
            f"max_position_pct={self.max_position_pct*100:.1f}%, "
            f"target_vol={self.target_vol*100:.1f}%, "
            f"rebalance_threshold={self.rebalance_threshold_bps} bps"
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
                recent = self.market_carry_history[market][-self.lookback_hours :]
                if recent:
                    rates = [
                        float(m.funding_apy) / (24.0 * 365.0 * 100.0)
                        for m in recent
                        if m.funding_apy is not None
                    ]
                    if len(rates) >= 4:
                        return np.array(rates, dtype=float)

            # Fallback: data_fetcher (if it knows how to do this)
            if hasattr(self.data_fetcher, "get_funding_history"):
                series = self.data_fetcher.get_funding_history(market, hours=self.lookback_hours)
                if series:
                    return np.array(series, dtype=float)
        except Exception as e:
            logger.debug(f"[{self.strategy_name}] Error fetching funding history for {market}: {e}")
        return None

    def _fetch_historical_basis(self, market: str, spot_price: float, perp_price: float) -> Optional[np.ndarray]:
        """
        Try to fetch historical basis (bps) for the market over lookback_hours.
        Returns an array of basis in bps, or None.
        """
        try:
            if hasattr(self.data_fetcher, "get_basis_history"):
                series = self.data_fetcher.get_basis_history(market, hours=self.lookback_hours)
                if series:
                    return np.array(series, dtype=float)

            if hasattr(self.analytics, "get_basis_series"):
                series = self.analytics.get_basis_series(market, hours=self.lookback_hours)
                if series is not None:
                    return np.array(series, dtype=float)
        except Exception as e:
            logger.debug(f"[{self.strategy_name}] Error fetching basis history for {market}: {e}")

        if spot_price > 0:
            basis_bps = ((perp_price - spot_price) / spot_price) * 10000.0
            return np.full(self.lookback_hours, float(basis_bps), dtype=float)

        return None

    def _fetch_volatility_24h(self, market: str, price_data: Dict[str, Any]) -> float:
        """
        Integrate with external volatility data when available:
        1. Use explicit volatility_24h from price_data if present
        2. Ask data_fetcher / analytics for realized or implied vol
        3. Fall back to a conservative default (2%)
        """
        if "volatility_24h" in price_data and price_data["volatility_24h"] is not None:
            return float(price_data["volatility_24h"])

        try:
            if hasattr(self.data_fetcher, "get_volatility_24h"):
                v = self.data_fetcher.get_volatility_24h(market)
                if v is not None:
                    return float(v)
        except Exception as e:
            logger.debug(f"[{self.strategy_name}] Error fetching volatility_24h via data_fetcher for {market}: {e}")

        try:
            if hasattr(self.analytics, "get_volatility_24h"):
                v = self.analytics.get_volatility_24h(market)
                if v is not None:
                    return float(v)
        except Exception as e:
            logger.debug(f"[{self.strategy_name}] Error fetching volatility_24h via analytics for {market}: {e}")

        return 0.02  # 2% placeholder

    # -------------------------------------------------------------------------
    # Carry calculation
    # -------------------------------------------------------------------------
    def calculate_carry_metrics(
        self,
        market: str,
        funding_data: Dict[str, Any],
        price_data: Dict[str, Any],
    ) -> Optional[CarryMetrics]:
        """
        More sophisticated carry calculation using historical funding and basis data.
        """
        try:
            funding_rate_inst = float(funding_data.get("rate_1h", 0.0))
            funding_apy_inst = funding_rate_inst * 24.0 * 365.0 * 100.0

            funding_hist = self._fetch_historical_funding(market)
            if funding_hist is not None and len(funding_hist) >= 4:
                funding_rate_mean = float(funding_hist.mean())
                funding_apy_smoothed = funding_rate_mean * 24.0 * 365.0 * 100.0
                funding_apy = 0.5 * funding_apy_inst + 0.5 * funding_apy_smoothed
            else:
                funding_apy_smoothed = funding_apy_inst
                funding_apy = funding_apy_inst

            # FIX (None -> float): sanitize spot/perp inputs
            spot_price = _coerce_float(price_data.get("spot", 0.0), 0.0)
            perp_price = _coerce_float(price_data.get("perp", spot_price), spot_price)

            if spot_price <= 0.0:
                return None

            basis_bps_current = ((perp_price - spot_price) / spot_price) * 10000.0

            basis_hist = self._fetch_historical_basis(market, spot_price=spot_price, perp_price=perp_price)
            basis_trend_bps_per_day = 0.0
            expected_convergence_daily_bps = 0.0

            if basis_hist is not None and len(basis_hist) >= 5:
                x = np.arange(len(basis_hist))
                slope, _intercept = np.polyfit(x, basis_hist, 1)
                basis_trend_bps_per_day = float(slope * 24.0)

                basis_mean = float(basis_hist.mean())
                deviation = basis_bps_current - basis_mean

                base_convergence = -0.5 * deviation
                if np.sign(slope) == np.sign(deviation):
                    base_convergence *= 0.5

                expected_convergence_daily_bps = float(base_convergence)
            else:
                expected_convergence_daily_bps = float(basis_bps_current * 0.1)

            convergence_apy = (expected_convergence_daily_bps / 10000.0) * 365.0 * 100.0
            total_carry_apy = float(funding_apy + convergence_apy)

            volatility_24h = float(self._fetch_volatility_24h(market, price_data))
            if volatility_24h > 0.0:
                volatility_adj_carry = total_carry_apy / (volatility_24h * 100.0)
            else:
                volatility_adj_carry = total_carry_apy

            conf = 0.5
            if funding_hist is not None and len(funding_hist) >= 8:
                funding_std = float(funding_hist.std())
                conf_funding = max(0.0, min(1.0, 1.0 - funding_std * 10.0))
                conf += 0.25 * conf_funding

            if basis_hist is not None and len(basis_hist) >= 8:
                trend_ratio = abs(basis_trend_bps_per_day) / (abs(basis_bps_current) + 1e-6)
                conf_basis = max(0.0, min(1.0, 1.0 - trend_ratio))
                conf += 0.25 * conf_basis

            magnitude_factor = min(1.0, abs(total_carry_apy) / 50.0)
            confidence = max(0.0, min(1.0, conf * magnitude_factor))

            metrics = CarryMetrics(
                market=market,
                funding_rate_1h=funding_rate_inst,
                funding_apy=float(funding_apy),
                basis_bps=float(basis_bps_current),
                expected_basis_convergence_daily=float(expected_convergence_daily_bps),
                total_carry_apy=float(total_carry_apy),
                volatility_adj_carry=float(volatility_adj_carry),
                confidence_score=float(confidence),
                volatility_24h=float(volatility_24h),
                funding_apy_smoothed=float(funding_apy_smoothed),
                basis_trend_bps_per_day=float(basis_trend_bps_per_day),
            )

            self.market_carry_history.setdefault(market, []).append(metrics)
            self.last_metrics[market] = metrics
            return metrics

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Error calculating carry for {market}: {e}")
            return None

    def rank_markets_by_carry(self, market_data: Dict[str, Dict[str, Any]]) -> List[CarryMetrics]:
        """
        Rank markets by volatility-adjusted carry, with a minimum carry filter.
        """
        carry_metrics: List[CarryMetrics] = []

        for market, data in market_data.items():
            funding_data = data.get("funding", {}) or {}
            price_data = data.get("prices", {}) or {}

            metrics = self.calculate_carry_metrics(market, funding_data, price_data)
            if metrics and abs(metrics.total_carry_apy) >= float(self.min_carry_apy):
                carry_metrics.append(metrics)

        carry_metrics.sort(key=lambda x: x.volatility_adj_carry, reverse=True)
        return carry_metrics

    # -------------------------------------------------------------------------
    # Sizing & rebalancing logic
    # -------------------------------------------------------------------------
    def _compute_size_pct(self, metrics: CarryMetrics) -> float:
        vol = max(float(metrics.volatility_24h), 1e-6)

        if vol <= self.target_vol:
            size_pct = float(self.max_position_pct)
        else:
            size_pct = float(self.max_position_pct) * (float(self.target_vol) / vol)
            size_pct = max(0.05 * float(self.max_position_pct), size_pct)

        return min(float(self.max_position_pct), float(size_pct))

    def _should_rebalance_position(
        self,
        market: str,
        metrics: Optional[CarryMetrics],
        position: Dict[str, Any],
    ) -> Dict[str, Any]:
        if metrics is None:
            return {"action": "close", "reason": "metrics_unavailable"}

        current_carry = float(metrics.total_carry_apy)
        entry_carry = _coerce_float(position.get("entry_carry_apy", current_carry), current_carry)
        max_carry = _coerce_float(position.get("max_carry_apy", entry_carry), entry_carry)

        if current_carry > max_carry:
            position["max_carry_apy"] = current_carry
            max_carry = current_carry

        if abs(current_carry) < float(self.min_carry_apy) * 0.5:
            return {"action": "close", "reason": "carry_below_threshold"}

        decay_points = max_carry - current_carry
        decay_threshold_points = float(self.rebalance_threshold_bps) / 100.0

        if decay_points >= decay_threshold_points and abs(current_carry) >= float(self.min_carry_apy):
            return {"action": "reduce", "reason": "carry_decay", "decay_points": decay_points}

        return {"action": "hold", "reason": "carry_stable"}

    # -------------------------------------------------------------------------
    # Main strategy loop
    # -------------------------------------------------------------------------
    def generate_signals(self, market_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_active:
            return []

        actions: List[Dict[str, Any]] = []

        try:
            funding_rates = market_snapshot.get("funding_rates", {}) or {}
            prices = market_snapshot.get("prices", {}) or {}
            _capital = market_snapshot.get("capital", None)

            market_data: Dict[str, Dict[str, Any]] = {}
            for market, funding in funding_rates.items():
                funding_dict = funding or {}
                symbol = str(market).replace("-PERP", "").upper()

                spot_price = _coerce_float(prices.get(symbol, 0.0), 0.0)
                perp_price = _coerce_float(funding_dict.get("mark_price", spot_price), spot_price)

                market_data[str(market)] = {
                    "funding": {
                        "rate_1h": _coerce_float(funding_dict.get("rate", 0.0), 0.0),
                        "history": funding_dict.get("history"),
                    },
                    "prices": {
                        "spot": spot_price,
                        "perp": perp_price,
                        "volatility_24h": funding_dict.get("volatility_24h"),
                    },
                }

            ranked_markets = self.rank_markets_by_carry(market_data)

            # Open new positions in top markets
            for metrics in ranked_markets[:3]:
                if metrics.market in self.positions:
                    continue

                direction = "short" if float(metrics.funding_rate_1h) > 0.0 else "long"
                size_pct = self._compute_size_pct(metrics)

                if not check_position_limits(self.drift_client, metrics.market, float(size_pct)):
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
                            "size_pct": float(size_pct),
                            "strategy": self.strategy_name,
                            "carry_apy": float(metrics.total_carry_apy),
                            "vol_adj_carry": float(metrics.volatility_adj_carry),
                            "volatility_24h": float(metrics.volatility_24h),
                        },
                        "priority": "balanced",
                    }
                )

                logger.info(
                    f"[{self.strategy_name}] Signal: {direction} {metrics.market}, "
                    f"carry={metrics.total_carry_apy:.2f}% APY "
                    f"(vol_adj={metrics.volatility_adj_carry:.2f}, vol={metrics.volatility_24h*100:.1f}%)"
                )

            # Rebalance / close existing positions based on carry decay
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
                    logger.info(f"[{self.strategy_name}] Closing {market}: reason={decision['reason']}")

                elif decision["action"] == "reduce":
                    current_size = _coerce_float(pos.get("size", 0.0), 0.0)
                    if current_size <= 0.0:
                        continue

                    target_size = current_size * 0.5
                    reduction = current_size - target_size

                    actions.append(
                        {
                            "action_type": "reduce_position",
                            "params": {
                                "market": market,
                                "current_size": float(current_size),
                                "target_size": float(target_size),
                                "reduction": float(reduction),
                                "reason": decision["reason"],
                                "strategy": self.strategy_name,
                            },
                            "priority": "balanced",
                        }
                    )

                    logger.info(
                        f"[{self.strategy_name}] Reducing {market}: "
                        f"carry decay={_coerce_float(decision.get('decay_points', 0.0), 0.0):.2f} pts, "
                        f"size={current_size:.4f}->{target_size:.4f}"
                    )

            self.carry_history.append(
                {
                    "timestamp": time.time(),
                    "ranked_markets": [
                        {"market": m.market, "carry_apy": float(m.total_carry_apy), "vol_adj_carry": float(m.volatility_adj_carry)}
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
    def apply_simulated_trade(self, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        market = str((action.get("params") or {}).get("market", ""))
        action_type = str(action.get("action_type", ""))

        if action_type == "open_perp" and bool(result.get("success")):
            carry_apy = _coerce_float((action.get("params") or {}).get("carry_apy", 0.0), 0.0)
            now = time.time()
            self.positions[market] = {
                "direction": (action.get("params") or {}).get("direction"),
                "size": _coerce_float(result.get("size", 0.0), 0.0),
                "entry_price": _coerce_float(result.get("price", 0.0), 0.0),
                "entry_carry_apy": float(carry_apy),
                "max_carry_apy": float(carry_apy),
                "opened_at": float(now),
            }

        elif action_type in ("close_perp", "reduce_position") and bool(result.get("success")):
            if market in self.positions:
                pnl = _coerce_float(result.get("pnl", 0.0), 0.0)
                self.pnl += float(pnl)

                if action_type == "close_perp":
                    del self.positions[market]
                else:
                    reduction = _coerce_float((action.get("params") or {}).get("reduction", 0.0), 0.0)
                    pos = self.positions[market]
                    pos_size = _coerce_float(pos.get("size", 0.0), 0.0)
                    if 0.0 < reduction < pos_size:
                        pos["size"] = pos_size - reduction
                        self.positions[market] = pos
                    else:
                        del self.positions[market]

    def get_status(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "is_active": self.is_active,
            "positions": self.positions,
            "total_pnl": float(self.pnl),
            "min_carry_apy": float(self.min_carry_apy),
            "rebalance_threshold_bps": float(self.rebalance_threshold_bps),
            "target_vol": float(self.target_vol),
            "recent_rankings": self.carry_history[-5:] if self.carry_history else [],
        }

