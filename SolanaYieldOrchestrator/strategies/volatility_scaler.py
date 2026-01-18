"""
Volatility Scaler Strategy
Dynamically adjusts position sizes based on rolling realized volatility.
Higher volatility -> smaller positions, lower volatility -> larger positions.

Features:
- GARCH(1,1)-based volatility forecasting
- Regime detection for volatility clustering
- Integration with options IV data when available
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd  # kept for downstream integrations / optional analytics usage

from config import Config  # noqa: F401 (reserved for future config-driven tuning)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _coerce_float(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class VolatilityMetrics:
    market: str
    realized_vol_1h: float
    realized_vol_24h: float
    realized_vol_7d: float
    vol_percentile: float
    recommended_size_multiplier: float
    vol_regime: str
    garch_vol_forecast_24h: float
    iv_vol_annualized: Optional[float]
    effective_vol_24h: float


class VolatilityScaler:
    def __init__(
        self,
        base_position_size: float = 1000.0,
        target_vol: float = 0.20,
        max_vol_multiplier: float = 2.0,
        min_vol_multiplier: float = 0.25,
        lookback_periods: int = 168,
        update_frequency_sec: int = 3600,
        # GARCH(1,1) parameters (simple defaults; can be tuned or made config-driven)
        garch_alpha: float = 0.05,
        garch_beta: float = 0.90,
        garch_omega: float = 1e-6,
        # Regime / clustering config
        high_vol_percentile: float = 80.0,
        low_vol_percentile: float = 20.0,
        cluster_window: int = 10,
        cluster_min_hits: int = 3,
        # IV integration
        iv_weight: float = 0.5,
    ):
        self.base_position_size = float(base_position_size)
        self.target_vol = float(target_vol)
        self.max_vol_multiplier = float(max_vol_multiplier)
        self.min_vol_multiplier = float(min_vol_multiplier)
        self.lookback_periods = int(lookback_periods)
        self.update_frequency_sec = int(update_frequency_sec)

        self.garch_alpha = float(garch_alpha)
        self.garch_beta = float(garch_beta)
        self.garch_omega = float(garch_omega)

        self.high_vol_percentile = float(high_vol_percentile)
        self.low_vol_percentile = float(low_vol_percentile)
        self.cluster_window = int(cluster_window)
        self.cluster_min_hits = int(cluster_min_hits)

        self.iv_weight = float(iv_weight)

        self.strategy_name = "volatility_scaler"
        self.is_active = True

        self.price_history: Dict[str, deque] = {}
        self.vol_history: Dict[str, List[Dict[str, Any]]] = {}
        self.current_multipliers: Dict[str, float] = {}
        self.last_update: float = 0.0

        # Optional inputs / state
        self.implied_vols: Dict[str, float] = {}
        self.garch_forecasts: Dict[str, float] = {}

        logger.info(
            f"[{self.strategy_name}] Initialized with target_vol={self.target_vol*100:.1f}%, "
            f"multiplier_range=[{self.min_vol_multiplier:.2f}, {self.max_vol_multiplier:.2f}], "
            f"GARCH(alpha={self.garch_alpha}, beta={self.garch_beta}, omega={self.garch_omega}), "
            f"iv_weight={self.iv_weight:.2f}"
        )

    # -------------------------------------------------------------------------
    # Data updates
    # -------------------------------------------------------------------------
    def update_price(self, market: str, price: float, timestamp: Optional[float] = None) -> None:
        # FIX: timestamp is Optional[float] (not float with None default)
        if market not in self.price_history:
            self.price_history[market] = deque(maxlen=self.lookback_periods)

        ts = time.time() if timestamp is None else float(timestamp)
        self.price_history[market].append({"price": float(price), "timestamp": ts})

    def update_implied_vol(self, market: str, iv_annualized: float) -> None:
        """
        Integrate options IV data when available.

        iv_annualized: annualized implied volatility (e.g., 0.80 for 80%).
        """
        self.implied_vols[market] = float(iv_annualized)

    # -------------------------------------------------------------------------
    # Realized / GARCH volatility
    # -------------------------------------------------------------------------
    def _get_returns(self, market: str, periods: Optional[int] = None) -> Optional[np.ndarray]:
        if market not in self.price_history or len(self.price_history[market]) < 2:
            return None

        prices = [float(p["price"]) for p in self.price_history[market]]
        if periods is not None:
            prices = prices[-int(periods) :]
        if len(prices) < 2:
            return None

        returns = np.diff(np.log(prices))
        return returns

    def calculate_realized_volatility(self, market: str, periods: int = 24) -> float:
        """
        Realized annualized volatility based on log returns over the last `periods` points.
        Assumes returns are approximately hourly.
        """
        returns = self._get_returns(market, periods=int(periods))
        if returns is None or len(returns) < 2:
            return float(self.target_vol)

        hourly_vol = float(np.std(returns))
        annualized_vol = hourly_vol * float(np.sqrt(24.0 * 365.0))
        return float(annualized_vol)

    def forecast_garch_volatility(self, market: str, periods: int = 24) -> Optional[float]:
        """
        Simple GARCH(1,1)-based volatility forecasting on log returns.

        Returns annualized forecast volatility. If insufficient data, returns None.
        """
        returns = self._get_returns(market)
        if returns is None or len(returns) < max(10, int(periods) // 2):
            return None

        r2 = returns**2
        var_init = float(r2.mean()) if float(r2.mean()) > 0.0 else float(np.var(returns))
        if var_init <= 0.0:
            var_init = 1e-8

        sigma2 = np.zeros_like(returns)
        sigma2[0] = var_init

        alpha = float(self.garch_alpha)
        beta = float(self.garch_beta)
        omega = float(self.garch_omega)

        for t in range(1, len(returns)):
            sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]

        var_forecast_1h = float(omega + alpha * r2[-1] + beta * sigma2[-1])
        var_forecast_1h = max(var_forecast_1h, 1e-12)

        # Scale to horizon by sqrt(h)
        vol_h = float(np.sqrt(var_forecast_1h) * np.sqrt(float(periods)))
        annualized_vol = float(vol_h * np.sqrt(365.0))

        self.garch_forecasts[market] = annualized_vol
        return annualized_vol

    # -------------------------------------------------------------------------
    # Metrics / regimes / multipliers
    # -------------------------------------------------------------------------
    def _detect_regime_with_clustering(self, market: str, base_regime: str, vol_percentile: float) -> str:
        history = self.vol_history.get(market, [])
        window = history[-self.cluster_window :] if self.cluster_window > 0 else history
        high_hits = sum(1 for h in window if str(h.get("regime", "")).startswith("high_vol"))
        low_hits = sum(1 for h in window if str(h.get("regime", "")).startswith("low_vol"))

        regime = base_regime
        if base_regime.startswith("high_vol") and high_hits >= self.cluster_min_hits:
            regime = "high_vol_clustered"
        elif base_regime.startswith("low_vol") and low_hits >= self.cluster_min_hits:
            regime = "low_vol_clustered"

        return regime

    def calculate_vol_metrics(self, market: str) -> VolatilityMetrics:
        has_history = len(self.price_history.get(market, [])) >= 2

        vol_1h = self.calculate_realized_volatility(market, periods=2) if has_history else float(self.target_vol)
        vol_24h = self.calculate_realized_volatility(market, periods=24)
        vol_7d = self.calculate_realized_volatility(market, periods=168)

        if market in self.vol_history and len(self.vol_history[market]) >= 10:
            historical_vols = [float(v.get("vol_24h", vol_24h)) for v in self.vol_history[market][-100:]]
            if historical_vols:
                vol_percentile = (sum(1 for v in historical_vols if v < vol_24h) / len(historical_vols)) * 100.0
            else:
                vol_percentile = 50.0
        else:
            vol_percentile = 50.0

        garch_vol_24h = self.forecast_garch_volatility(market, periods=24)
        if garch_vol_24h is None:
            garch_vol_24h = float(vol_24h)

        iv_vol = self.implied_vols.get(market)  # Optional[float]

        if iv_vol is not None:
            effective_vol_24h = float(self.iv_weight) * float(iv_vol) + (1.0 - float(self.iv_weight)) * float(garch_vol_24h)
        else:
            effective_vol_24h = float(garch_vol_24h)

        # Volatility-scaling multiplier
        if effective_vol_24h > 0.0:
            size_multiplier = float(self.target_vol) / float(effective_vol_24h)
        else:
            size_multiplier = 1.0
        size_multiplier = max(float(self.min_vol_multiplier), min(float(self.max_vol_multiplier), float(size_multiplier)))

        if vol_percentile >= float(self.high_vol_percentile):
            base_regime = "high_vol"
        elif vol_percentile <= float(self.low_vol_percentile):
            base_regime = "low_vol"
        else:
            base_regime = "normal"

        regime = self._detect_regime_with_clustering(market, base_regime, vol_percentile)

        metrics = VolatilityMetrics(
            market=market,
            realized_vol_1h=float(vol_1h),
            realized_vol_24h=float(vol_24h),
            realized_vol_7d=float(vol_7d),
            vol_percentile=float(vol_percentile),
            recommended_size_multiplier=float(size_multiplier),
            vol_regime=str(regime),
            garch_vol_forecast_24h=float(garch_vol_24h),
            iv_vol_annualized=iv_vol,
            effective_vol_24h=float(effective_vol_24h),
        )

        self.update_vol_history(market, metrics)
        return metrics

    def get_position_size(self, market: str, base_size: Optional[float] = None) -> float:
        # FIX: base_size Optional[float] (not float with None default)
        base = float(self.base_position_size) if base_size is None else float(base_size)

        metrics = self.calculate_vol_metrics(market)
        self.current_multipliers[market] = float(metrics.recommended_size_multiplier)

        adjusted_size = base * float(metrics.recommended_size_multiplier)

        # FIX: avoid multiplying None (iv_vol_annualized can be None)
        iv_str = f"{float(metrics.iv_vol_annualized) * 100:.1f}%" if metrics.iv_vol_annualized is not None else "N/A"

        logger.debug(
            f"[{self.strategy_name}] {market}: "
            f"24h_realized_vol={metrics.realized_vol_24h*100:.1f}%, "
            f"garch_24h={metrics.garch_vol_forecast_24h*100:.1f}%, "
            f"iv={iv_str}, "
            f"effective_vol_24h={metrics.effective_vol_24h*100:.1f}%, "
            f"multiplier={metrics.recommended_size_multiplier:.2f}, "
            f"size={base:.2f}->{adjusted_size:.2f}, "
            f"regime={metrics.vol_regime}"
        )

        return float(adjusted_size)

    def adjust_existing_positions(self, positions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        adjustments: List[Dict[str, Any]] = []

        for market, position in positions.items():
            current_size = _coerce_float((position or {}).get("size", 0.0), 0.0)
            if current_size == 0.0:
                continue

            metrics = self.calculate_vol_metrics(market)
            target_size = self.get_position_size(market, self.base_position_size)

            size_diff_pct = abs(current_size - target_size) / max(current_size, 1e-8) * 100.0

            if size_diff_pct > 20.0:
                if current_size > target_size:
                    adjustments.append(
                        {
                            "action_type": "reduce_position",
                            "params": {
                                "market": market,
                                "current_size": float(current_size),
                                "target_size": float(target_size),
                                "reduction": float(current_size - target_size),
                                "reason": f"vol_increase ({metrics.vol_regime})",
                                "strategy": self.strategy_name,
                            },
                            "priority": "balanced",
                        }
                    )
                else:
                    adjustments.append(
                        {
                            "action_type": "increase_position",
                            "params": {
                                "market": market,
                                "current_size": float(current_size),
                                "target_size": float(target_size),
                                "increase": float(target_size - current_size),
                                "reason": f"vol_decrease ({metrics.vol_regime})",
                                "strategy": self.strategy_name,
                            },
                            "priority": "balanced",
                        }
                    )

        return adjustments

    def update_vol_history(self, market: str, metrics: VolatilityMetrics) -> None:
        if market not in self.vol_history:
            self.vol_history[market] = []

        self.vol_history[market].append(
            {
                "timestamp": time.time(),
                "vol_1h": float(metrics.realized_vol_1h),
                "vol_24h": float(metrics.realized_vol_24h),
                "vol_7d": float(metrics.realized_vol_7d),
                "regime": str(metrics.vol_regime),
                "garch_vol_24h": float(metrics.garch_vol_forecast_24h),
                "iv_vol": metrics.iv_vol_annualized,
                "effective_vol_24h": float(metrics.effective_vol_24h),
            }
        )

        if len(self.vol_history[market]) > 1000:
            self.vol_history[market] = self.vol_history[market][-500:]

    def get_all_vol_metrics(self) -> Dict[str, Dict[str, Any]]:
        metrics_out: Dict[str, Dict[str, Any]] = {}
        for market in self.price_history.keys():
            m = self.calculate_vol_metrics(market)
            metrics_out[market] = {
                "realized_vol_1h": float(m.realized_vol_1h),
                "realized_vol_24h": float(m.realized_vol_24h),
                "realized_vol_7d": float(m.realized_vol_7d),
                "vol_percentile": float(m.vol_percentile),
                "size_multiplier": float(m.recommended_size_multiplier),
                "regime": str(m.vol_regime),
                "garch_vol_forecast_24h": float(m.garch_vol_forecast_24h),
                "iv_vol_annualized": m.iv_vol_annualized,
                "effective_vol_24h": float(m.effective_vol_24h),
            }
        return metrics_out

    def get_status(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "is_active": bool(self.is_active),
            "target_vol": float(self.target_vol),
            "current_multipliers": self.current_multipliers,
            "vol_metrics": self.get_all_vol_metrics(),
            "markets_tracked": list(self.price_history.keys()),
            "garch_params": {
                "alpha": float(self.garch_alpha),
                "beta": float(self.garch_beta),
                "omega": float(self.garch_omega),
            },
            "iv_weight": float(self.iv_weight),
        }

