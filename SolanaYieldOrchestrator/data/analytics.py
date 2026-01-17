from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import Config  # noqa: F401  (kept in case used elsewhere in the package)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BasisAnalysis:
    spot_price: float
    perp_price: float
    basis_bps: float
    basis_pct: float
    annualized_apy: float
    is_contango: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class FundingAnalysis:
    market: str
    current_rate: float
    rate_1h: float
    rate_8h: float
    estimated_apy: float
    direction: str


@dataclass
class PortfolioMetrics:
    total_nav: float
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    weekly_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PnLComponents:
    """Tracks PnL broken down by source: funding, basis, and fees."""
    funding_pnl: float = 0.0
    basis_pnl: float = 0.0
    fees_pnl: float = 0.0
    mark_to_market_pnl: float = 0.0

    @property
    def total(self) -> float:
        return float(self.funding_pnl + self.basis_pnl - self.fees_pnl + self.mark_to_market_pnl)

    @property
    def yield_only(self) -> float:
        """PnL from yield sources only (funding + basis)."""
        return float(self.funding_pnl + self.basis_pnl)

    def to_dict(self) -> Dict[str, float]:
        # Ensure plain Python floats (not numpy floating)
        return {
            "funding": float(self.funding_pnl),
            "basis": float(self.basis_pnl),
            "fees": float(self.fees_pnl),
            "mark_to_market": float(self.mark_to_market_pnl),
            "yield_only": float(self.yield_only),
            "total": float(self.total),
        }


class Analytics:
    def __init__(self):
        self._nav_history: List[Dict[str, Any]] = []
        self._pnl_history: List[Dict[str, Any]] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._pnl_components_history: List[Dict[str, Any]] = []
        self._exposure_history: List[Dict[str, Any]] = []
        self._initial_nav = 0.0
        self._pnl_components = PnLComponents()

    def calculate_basis(self, spot_price: float, perp_price: float, funding_rate: float = 0.0) -> BasisAnalysis:
        if spot_price <= 0:
            return BasisAnalysis(
                spot_price=float(spot_price),
                perp_price=float(perp_price),
                basis_bps=0.0,
                basis_pct=0.0,
                annualized_apy=0.0,
                is_contango=False,
            )

        basis_pct = ((perp_price - spot_price) / spot_price) * 100.0
        basis_bps = basis_pct * 100.0

        funding_apy = float(funding_rate * 24.0 * 365.0 * 100.0) if funding_rate else 0.0
        basis_apy = float(basis_pct * 365.0)
        annualized_apy = float(funding_apy + basis_apy)

        return BasisAnalysis(
            spot_price=float(spot_price),
            perp_price=float(perp_price),
            basis_bps=float(basis_bps),
            basis_pct=float(basis_pct),
            annualized_apy=float(annualized_apy),
            is_contango=perp_price > spot_price,
        )

    def analyze_funding(self, funding_data: Dict[str, Any]) -> List[FundingAnalysis]:
        analyses: List[FundingAnalysis] = []

        for market, data in funding_data.items():
            if isinstance(data, dict):
                rate_any = data.get("rate", data.get("funding_rate", 0.0))
                apy_any = data.get("apy", data.get("funding_apy", 0.0))
                rate = float(rate_any or 0.0)
                apy = float(apy_any or 0.0)
            else:
                rate = float(getattr(data, "funding_rate", 0.0) or 0.0)
                apy = float(getattr(data, "funding_apy", 0.0) or 0.0)

            direction = "long_pays" if rate > 0 else "short_pays" if rate < 0 else "neutral"

            analyses.append(
                FundingAnalysis(
                    market=str(market),
                    current_rate=float(rate),
                    rate_1h=float(rate),
                    rate_8h=float(rate) * 8.0,
                    estimated_apy=float(apy) if apy else float(rate) * 24.0 * 365.0 * 100.0,
                    direction=direction,
                )
            )

        return sorted(analyses, key=lambda x: abs(x.estimated_apy), reverse=True)

    def rank_markets_by_funding(self, funding_analyses: List[FundingAnalysis], min_apy: float = 0.0) -> List[FundingAnalysis]:
        filtered = [f for f in funding_analyses if abs(f.estimated_apy) >= float(min_apy)]
        return sorted(filtered, key=lambda x: abs(x.estimated_apy), reverse=True)

    def record_nav(self, nav: float, positions: Optional[Dict[str, float]] = None) -> None:
        if self._initial_nav == 0.0:
            self._initial_nav = float(nav)

        self._nav_history.append(
            {
                "timestamp": time.time(),
                "nav": float(nav),
                "positions": dict(positions) if positions is not None else {},
            }
        )

        if len(self._nav_history) > 10000:
            self._nav_history = self._nav_history[-5000:]

    def record_pnl(self, pnl: float, strategy: str = "total") -> None:
        self._pnl_history.append({"timestamp": time.time(), "pnl": float(pnl), "strategy": str(strategy)})

        if len(self._pnl_history) > 10000:
            self._pnl_history = self._pnl_history[-5000:]

    def record_trade(self, trade: Dict[str, Any]) -> None:
        trade["timestamp"] = float(trade.get("timestamp", time.time()) or time.time())
        self._trade_history.append(trade)

        if len(self._trade_history) > 10000:
            self._trade_history = self._trade_history[-5000:]

    def record_pnl_component(self, component_type: str, amount: float, market: Optional[str] = None) -> None:
        """Record a PnL contribution by type (funding, basis, fees, mark_to_market)."""
        amt = float(amount)

        if component_type == "funding":
            self._pnl_components.funding_pnl += amt
        elif component_type == "basis":
            self._pnl_components.basis_pnl += amt
        elif component_type == "fees":
            self._pnl_components.fees_pnl += abs(amt)
        elif component_type == "mark_to_market":
            self._pnl_components.mark_to_market_pnl += amt

        self._pnl_components_history.append(
            {
                "timestamp": time.time(),
                "component": str(component_type),
                "amount": amt,
                "market": str(market) if market is not None else None,
            }
        )

        if len(self._pnl_components_history) > 10000:
            self._pnl_components_history = self._pnl_components_history[-5000:]

    def record_exposure(self, exposures: Dict[str, float]) -> None:
        """Record current exposure by market."""
        self._exposure_history.append({"timestamp": time.time(), "exposures": dict(exposures)})

        if len(self._exposure_history) > 10000:
            self._exposure_history = self._exposure_history[-5000:]

    def get_pnl_components(self) -> Dict[str, float]:
        """Get current PnL breakdown by component."""
        return self._pnl_components.to_dict()

    def get_pnl_components_timeseries(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get PnL components over time for charting."""
        cutoff = time.time() - (int(hours) * 3600)
        return [h for h in self._pnl_components_history if float(h["timestamp"]) >= cutoff]

    def get_max_exposure_by_market(self) -> Dict[str, float]:
        """Calculate maximum exposure seen per market over history."""
        max_exposures: Dict[str, float] = {}

        for entry in self._exposure_history:
            exposures = entry.get("exposures", {})
            if not isinstance(exposures, dict):
                continue
            for market, exposure in exposures.items():
                try:
                    exp_val = float(exposure)
                except (TypeError, ValueError):
                    continue
                current_max = max_exposures.get(str(market), 0.0)
                max_exposures[str(market)] = float(max(current_max, abs(exp_val)))

        return max_exposures

    def get_portfolio_metrics(self, current_nav: Optional[float] = None) -> PortfolioMetrics:
        if not self._nav_history:
            return PortfolioMetrics(
                total_nav=float(current_nav or 0.0),
                total_pnl=0.0,
                total_pnl_pct=0.0,
                daily_pnl=0.0,
                weekly_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
            )

        nav = float(current_nav) if current_nav is not None else float(self._nav_history[-1]["nav"])

        total_pnl = float(nav - self._initial_nav)
        total_pnl_pct = float((total_pnl / self._initial_nav * 100.0) if self._initial_nav > 0 else 0.0)

        navs = [float(h["nav"]) for h in self._nav_history]

        now = time.time()
        day_ago = now - 86400
        week_ago = now - 604800

        daily_navs = [float(h["nav"]) for h in self._nav_history if float(h["timestamp"]) >= day_ago]
        weekly_navs = [float(h["nav"]) for h in self._nav_history if float(h["timestamp"]) >= week_ago]

        daily_pnl = float(nav - daily_navs[0]) if daily_navs else 0.0
        weekly_pnl = float(nav - weekly_navs[0]) if weekly_navs else 0.0

        max_drawdown = float(self._calculate_max_drawdown(navs))

        if len(navs) > 1:
            returns = np.diff(np.array(navs, dtype=float)) / np.array(navs[:-1], dtype=float)
            vol = np.std(returns)
            volatility = float(vol * np.sqrt(365.0 * 24.0) * 100.0)
            mean_return = float(np.mean(returns))
            sharpe_ratio = float((mean_return / float(vol)) * np.sqrt(365.0 * 24.0)) if float(vol) > 0 else 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0

        return PortfolioMetrics(
            total_nav=float(nav),
            total_pnl=float(total_pnl),
            total_pnl_pct=float(total_pnl_pct),
            daily_pnl=float(daily_pnl),
            weekly_pnl=float(weekly_pnl),
            max_drawdown=float(max_drawdown),
            sharpe_ratio=float(sharpe_ratio),
            volatility=float(volatility),
        )

    def _calculate_max_drawdown(self, navs: List[float]) -> float:
        if not navs:
            return 0.0

        peak = float(navs[0])
        max_dd = 0.0

        for nav in navs:
            nav_f = float(nav)
            if nav_f > peak:
                peak = nav_f
            drawdown = float((peak - nav_f) / peak * 100.0) if peak > 0 else 0.0
            max_dd = float(max(max_dd, drawdown))

        return float(max_dd)

    def calculate_sortino_ratio(self, nav_series: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)."""
        if len(nav_series) < 2:
            return 0.0

        nav_arr = np.array(nav_series, dtype=float)
        returns = np.diff(nav_arr) / nav_arr[:-1]
        excess_returns = returns - (float(risk_free_rate) / float(periods_per_year))

        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0

        if downside_deviation <= 0:
            return 0.0

        mean_excess_return = float(np.mean(excess_returns))
        sortino = (mean_excess_return / downside_deviation) * np.sqrt(float(periods_per_year))

        return float(sortino)

    def calculate_nav_volatility(self, nav_series: List[float], periods_per_year: int = 8760) -> float:
        """Calculate annualized volatility of NAV returns."""
        if len(nav_series) < 2:
            return 0.0

        nav_arr = np.array(nav_series, dtype=float)
        returns = np.diff(nav_arr) / nav_arr[:-1]
        volatility = float(np.std(returns) * np.sqrt(float(periods_per_year)) * 100.0)

        return float(volatility)

    def calculate_extended_metrics(self, nav_series: Optional[List[float]] = None) -> Dict[str, Any]:
        """Calculate extended quant metrics including Sortino, volatility, max exposure."""
        if nav_series is None:
            nav_series = [float(h["nav"]) for h in self._nav_history]

        if len(nav_series) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "nav_volatility": 0.0,
                "max_drawdown": 0.0,
                "max_exposure_by_market": {},
                "pnl_components": self.get_pnl_components(),
            }

        nav_arr = np.array(nav_series, dtype=float)
        returns = np.diff(nav_arr) / nav_arr[:-1]

        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        sharpe = float((mean_return / std_return) * np.sqrt(8760.0)) if std_return > 0 else 0.0

        sortino = float(self.calculate_sortino_ratio(nav_series))
        nav_volatility = float(self.calculate_nav_volatility(nav_series))
        max_drawdown = float(self._calculate_max_drawdown(nav_series))
        max_exposure = self.get_max_exposure_by_market()
        pnl_components = self.get_pnl_components()

        return {
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "nav_volatility": float(nav_volatility),
            "max_drawdown": float(max_drawdown),
            "max_exposure_by_market": max_exposure,
            "pnl_components": pnl_components,
        }

    def get_nav_series(self, hours: int = 24) -> pd.DataFrame:
        if not self._nav_history:
            # Use pd.Index to satisfy pandas typing (Axes) and keep pyright happy
            return pd.DataFrame(columns=pd.Index(["timestamp", "nav"]))

        cutoff = time.time() - (int(hours) * 3600)
        filtered = [h for h in self._nav_history if float(h["timestamp"]) >= cutoff]

        df = pd.DataFrame(filtered)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.set_index("datetime")

        return df

    def get_pnl_by_strategy(self) -> Dict[str, float]:
        pnl_by_strategy: Dict[str, float] = {}

        for entry in self._pnl_history:
            strategy = str(entry.get("strategy", "unknown"))
            pnl = float(entry.get("pnl", 0.0) or 0.0)
            pnl_by_strategy[strategy] = float(pnl_by_strategy.get(strategy, 0.0) + pnl)

        return pnl_by_strategy

    def simulate_strategy_pnl(self, initial_capital: float, strategy_returns: List[float], days: int = 30) -> Dict[str, Any]:
        if not strategy_returns:
            return {
                "final_capital": float(initial_capital),
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "nav_series": [float(initial_capital)],
            }

        nav_series: List[float] = [float(initial_capital)]
        current_nav = float(initial_capital)

        for daily_return in strategy_returns[: int(days)]:
            current_nav *= (1.0 + float(daily_return) / 100.0)
            nav_series.append(float(current_nav))

        total_return = float((current_nav - initial_capital) / initial_capital * 100.0) if initial_capital != 0 else 0.0
        max_drawdown = float(self._calculate_max_drawdown(nav_series))

        returns = np.array(strategy_returns[: int(days)], dtype=float) / 100.0
        std_ret = float(np.std(returns))
        sharpe_ratio = float(np.mean(returns) / std_ret * np.sqrt(365.0)) if std_ret > 0 else 0.0

        return {
            "final_capital": float(current_nav),
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio),
            "nav_series": nav_series,
        }

    def estimate_fees(self, trade_size_usd: float, num_trades: int = 1, swap_fee_bps: float = 5, perp_fee_bps: float = 10) -> Dict[str, float]:
        trade_size = float(trade_size_usd)
        n = int(num_trades)

        swap_fees = trade_size * (float(swap_fee_bps) / 10000.0) * n
        perp_fees = trade_size * (float(perp_fee_bps) / 10000.0) * n
        total_fees = float(swap_fees + perp_fees)

        fee_drag_pct = float((total_fees / trade_size * 100.0) if trade_size > 0 else 0.0)

        return {
            "swap_fees": float(swap_fees),
            "perp_fees": float(perp_fees),
            "total_fees": float(total_fees),
            "fee_drag_pct": float(fee_drag_pct),
        }

    def run_scenario(
        self,
        positions: List[Dict[str, Any]],
        price_shocks: Dict[str, float],
        funding_shocks: Optional[Dict[str, float]] = None,
        current_nav: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Run a stress-test scenario on current positions.

        Args:
            positions: List of position dicts with market, size, side, entry_price, current_price
            price_shocks: Dict of symbol -> shock percentage (e.g., {"SOL": -0.2} for -20%)
            funding_shocks: Dict of market -> funding shock percentage
            current_nav: Current NAV before shocks

        Returns:
            Scenario results including shocked NAV, position PnL, liquidation warnings
        """
        funding_shocks_final: Dict[str, float] = dict(funding_shocks) if funding_shocks is not None else {}

        shocked_positions: List[Dict[str, Any]] = []
        total_pnl_change = 0.0
        liq_warnings: List[Dict[str, Any]] = []

        for pos in positions:
            market = str(pos.get("market", "") or "")
            size = float(abs(pos.get("size", 0.0) or 0.0))
            side = str(pos.get("side", "long") or "long")
            entry_price = float(pos.get("avg_entry_price", pos.get("entry_price", 0.0)) or 0.0)
            current_price = float(pos.get("current_price", entry_price) or entry_price)
            margin_used = float(pos.get("margin_used", 0.0) or 0.0)

            symbol = market.replace("-PERP", "").replace("/USDC", "").replace("/USD", "").upper()
            shock_pct = float(price_shocks.get(symbol, 0.0) or 0.0)

            shocked_price = float(current_price * (1.0 + shock_pct))

            if side.lower() in ["long", "buy"]:
                pnl_before = float(size * (current_price - entry_price))
                pnl_after = float(size * (shocked_price - entry_price))
            else:
                pnl_before = float(size * (entry_price - current_price))
                pnl_after = float(size * (entry_price - shocked_price))

            pnl_change = float(pnl_after - pnl_before)
            total_pnl_change += pnl_change

            notional = float(size * shocked_price)
            margin_ratio = float((margin_used + pnl_after) / notional) if notional > 0 else 0.0

            if side.lower() in ["long", "buy"]:
                liq_price = float(entry_price * (1.0 - (margin_used / notional) + 0.0625)) if notional > 0 else 0.0
            else:
                liq_price = float(entry_price * (1.0 + (margin_used / notional) - 0.0625)) if notional > 0 else 0.0

            if shocked_price > 0:
                if side.lower() in ["long", "buy"]:
                    dist_to_liq = float((shocked_price - liq_price) / shocked_price * 100.0)
                else:
                    dist_to_liq = float((liq_price - shocked_price) / shocked_price * 100.0)
            else:
                dist_to_liq = 0.0

            shocked_pos = {
                "market": market,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "current_price": current_price,
                "shocked_price": shocked_price,
                "shock_pct": shock_pct * 100.0,
                "pnl_before": pnl_before,
                "pnl_after": pnl_after,
                "pnl_change": pnl_change,
                "margin_ratio": margin_ratio,
                "liquidation_price": liq_price,
                "distance_to_liq_pct": dist_to_liq,
            }
            shocked_positions.append(shocked_pos)

            if margin_ratio < 0.10 and margin_used > 0:
                liq_warnings.append(
                    {
                        "market": market,
                        "side": side,
                        "margin_ratio": margin_ratio,
                        "distance_to_liq_pct": dist_to_liq,
                        "severity": "critical" if margin_ratio < 0.075 else "high",
                    }
                )

        shocked_nav = float(current_nav + total_pnl_change)
        nav_change_pct = float((total_pnl_change / current_nav * 100.0) if current_nav > 0 else 0.0)

        return {
            "current_nav": float(current_nav),
            "shocked_nav": float(shocked_nav),
            "nav_change": float(total_pnl_change),
            "nav_change_pct": float(nav_change_pct),
            "positions": shocked_positions,
            "liq_warnings": liq_warnings,
            "price_shocks_applied": dict(price_shocks),
            "funding_shocks_applied": funding_shocks_final,
        }

    def to_json_serializable(self) -> Dict[str, Any]:
        metrics = self.get_portfolio_metrics()
        extended_metrics = self.calculate_extended_metrics()

        return {
            "portfolio_metrics": {
                "total_nav": float(metrics.total_nav),
                "total_pnl": float(metrics.total_pnl),
                "total_pnl_pct": float(metrics.total_pnl_pct),
                "daily_pnl": float(metrics.daily_pnl),
                "weekly_pnl": float(metrics.weekly_pnl),
                "max_drawdown": float(metrics.max_drawdown),
                "sharpe_ratio": float(metrics.sharpe_ratio),
                "volatility": float(metrics.volatility),
            },
            "extended_metrics": extended_metrics,
            "pnl_by_strategy": self.get_pnl_by_strategy(),
            "pnl_breakdown": self.get_pnl_components(),
            "nav_history_length": int(len(self._nav_history)),
            "trade_count": int(len(self._trade_history)),
        }

    def reset(self) -> None:
        self._nav_history.clear()
        self._pnl_history.clear()
        self._trade_history.clear()
        self._pnl_components_history.clear()
        self._exposure_history.clear()
        self._initial_nav = 0.0
        self._pnl_components = PnLComponents()

    def calculate_perp_risk_metrics(self, perp_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not perp_positions:
            return {
                "positions": [],
                "aggregate": {
                    "total_notional": 0.0,
                    "total_margin": 0.0,
                    "weighted_avg_leverage": 0.0,
                    "overall_risk": "low",
                },
                "warnings": [],
            }

        position_risks: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        total_notional = 0.0
        total_margin = 0.0

        for pos in perp_positions:
            size = float(abs(pos.get("size", 0.0) or 0.0))
            entry_price = float(pos.get("avg_entry_price", pos.get("entry_price", 0.0)) or 0.0)
            current_price = float(pos.get("current_price", entry_price) or entry_price)
            margin = float(pos.get("margin_used", 0.0) or 0.0)
            side = str(pos.get("side", "long") or "long")
            leverage = float(pos.get("leverage", 1.0) or 1.0)

            if size == 0.0 or entry_price == 0.0:
                continue

            notional = float(size * current_price)
            total_notional += notional
            total_margin += margin

            maintenance_margin_ratio = 0.0625
            if side == "long":
                liq_price = float(entry_price * (1.0 - (margin / notional) + maintenance_margin_ratio)) if notional > 0 else 0.0
                distance_to_liq = float(((current_price - liq_price) / current_price) * 100.0) if current_price > 0 else 0.0
            else:
                liq_price = float(entry_price * (1.0 + (margin / notional) - maintenance_margin_ratio)) if notional > 0 else 0.0
                distance_to_liq = float(((liq_price - current_price) / current_price) * 100.0) if current_price > 0 else 0.0

            unrealized = float(pos.get("unrealized_pnl", 0.0) or 0.0)
            margin_ratio = float((margin + unrealized) / notional) if notional > 0 else 0.0

            if margin_ratio < 0.075:
                risk_level = "critical"
            elif margin_ratio < 0.10:
                risk_level = "high"
            elif margin_ratio < 0.20:
                risk_level = "medium"
            else:
                risk_level = "low"

            position_risks.append(
                {
                    "market": str(pos.get("market", "unknown")),
                    "side": side,
                    "size": size,
                    "leverage": leverage,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "notional": notional,
                    "margin_used": margin,
                    "margin_ratio": margin_ratio,
                    "liquidation_price": liq_price,
                    "distance_to_liq_pct": distance_to_liq,
                    "unrealized_pnl": unrealized,
                    "risk_level": risk_level,
                }
            )

            if risk_level in ["critical", "high"]:
                warnings.append(
                    {
                        "market": str(pos.get("market", "unknown")),
                        "type": "liquidation_warning",
                        "severity": risk_level,
                        "margin_ratio": margin_ratio,
                        "distance_to_liq_pct": distance_to_liq,
                        "recommended_action": "reduce_position" if risk_level == "critical" else "monitor",
                    }
                )

        weighted_leverage = float((total_notional / total_margin) if total_margin > 0 else 1.0)

        if any(p["risk_level"] == "critical" for p in position_risks):
            overall_risk = "critical"
        elif any(p["risk_level"] == "high" for p in position_risks):
            overall_risk = "high"
        elif any(p["risk_level"] == "medium" for p in position_risks):
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return {
            "positions": position_risks,
            "aggregate": {
                "total_notional": float(total_notional),
                "total_margin": float(total_margin),
                "weighted_avg_leverage": float(weighted_leverage),
                "overall_risk": overall_risk,
            },
            "warnings": warnings,
        }

    def calculate_partial_liquidation_ladder(
        self,
        position: Dict[str, Any],
        trim_levels: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if trim_levels is None:
            trim_levels = [0.10, 0.25, 0.50]

        size = float(abs(position.get("size", 0.0) or 0.0))
        margin = float(position.get("margin_used", 0.0) or 0.0)
        entry_price = float(position.get("avg_entry_price", 0.0) or 0.0)

        if size == 0.0 or margin == 0.0 or entry_price == 0.0:
            return []

        ladder: List[Dict[str, Any]] = []
        margin_thresholds: List[float] = [0.15, 0.10, 0.075]

        # Ensure trim_levels is long enough; zip will safely truncate otherwise.
        for i, (threshold, trim_pct) in enumerate(zip(margin_thresholds, trim_levels)):
            trim_size = float(size * float(trim_pct))
            remaining_size = float(size - trim_size)
            remaining_margin = float(margin * (1.0 - float(trim_pct)))

            notional = float(remaining_size * entry_price)
            new_margin_ratio = float(remaining_margin / notional) if notional > 0 else 0.0

            ladder.append(
                {
                    "level": int(i + 1),
                    "trigger_margin_ratio": float(threshold),
                    "trim_percentage": float(trim_pct) * 100.0,
                    "trim_size": trim_size,
                    "remaining_size": remaining_size,
                    "remaining_margin": remaining_margin,
                    "new_margin_ratio": new_margin_ratio,
                    "action": f"Trim {trim_pct*100:.0f}% of position",
                }
            )

        return ladder

    def calculate_strategy_breakdown(self, nav_history: List[Dict[str, Any]], strategy_pnl: Dict[str, float]) -> Dict[str, Any]:
        total_pnl = float(sum(float(v) for v in strategy_pnl.values()))

        breakdown: Dict[str, Any] = {}
        for strategy, pnl_val in strategy_pnl.items():
            pnl = float(pnl_val)
            contribution_pct = float((pnl / total_pnl * 100.0) if total_pnl != 0 else 0.0)
            breakdown[str(strategy)] = {
                "pnl": float(pnl),
                "contribution_pct": float(contribution_pct),
                "status": "profitable" if pnl > 0 else "loss" if pnl < 0 else "neutral",
            }

        best_strategy = max(strategy_pnl.items(), key=lambda x: float(x[1]))[0] if strategy_pnl else None
        worst_strategy = min(strategy_pnl.items(), key=lambda x: float(x[1]))[0] if strategy_pnl else None

        return {
            "by_strategy": breakdown,
            "total_pnl": float(total_pnl),
            "best_strategy": best_strategy,
            "worst_strategy": worst_strategy,
        }

    def calculate_returns_metrics(self, nav_series: List[float], periods_per_year: int = 8760) -> Dict[str, float]:
        if len(nav_series) < 2:
            return {
                "mean_return": 0.0,
                "std_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
            }

        nav_arr = np.array(nav_series, dtype=float)
        returns = np.diff(nav_arr) / nav_arr[:-1]

        mean_return = float(np.mean(returns) * float(periods_per_year) * 100.0)
        std_return = float(np.std(returns) * np.sqrt(float(periods_per_year)) * 100.0)

        std_r = float(np.std(returns))
        sharpe = float((np.mean(returns) / std_r) * np.sqrt(float(periods_per_year))) if std_r > 0 else 0.0

        downside_returns = returns[returns < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
        sortino = float((np.mean(returns) / downside_std) * np.sqrt(float(periods_per_year))) if downside_std > 0 else 0.0

        max_dd = float(self._calculate_max_drawdown([float(x) for x in nav_series]))
        calmar = float(mean_return / max_dd) if max_dd > 0 else 0.0

        return {
            "mean_return": float(mean_return),
            "std_return": float(std_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
        }
