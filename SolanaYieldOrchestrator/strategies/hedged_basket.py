"""
Hedged Basket Strategy
Creates a market-neutral basket of perpetual positions to capture yield
while maintaining near-zero net beta/delta exposure.

Features:
- Correlation-based basket construction
- Dynamic hedge ratio adjustment using beta estimates
- Volatility-scaled (risk-parity style) position sizing
- Portfolio-style capital allocation across long/short legs
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config  # noqa: F401 (kept for future config wiring)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BasketPosition:
    market: str
    side: str
    size: float
    weight: float
    delta_contribution: float
    funding_contribution: float
    beta: float = 1.0
    volatility: float = 0.0


@dataclass
class BasketMetrics:
    net_delta: float
    gross_exposure: float
    net_funding_apy: float
    hedge_ratio: float
    correlation_avg: float
    is_balanced: bool


class HedgedBasket:
    def __init__(
        self,
        target_delta: float = 0.0,
        delta_tolerance: float = 0.05,
        max_gross_exposure: float = 2.0,
        min_basket_size: int = 2,
        max_basket_size: int = 6,
        rebalance_threshold: float = 0.10,
        corr_threshold: float = 0.85,
    ):
        self.target_delta = target_delta
        self.delta_tolerance = delta_tolerance
        self.max_gross_exposure = max_gross_exposure
        self.min_basket_size = min_basket_size
        self.max_basket_size = max_basket_size
        self.rebalance_threshold = rebalance_threshold
        self.corr_threshold = corr_threshold

        self.strategy_name = "hedged_basket"
        self.is_active = True

        self.basket_positions: Dict[str, BasketPosition] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.price_history: Dict[str, List[float]] = {}
        self.volatility_map: Dict[str, float] = {}
        self.beta_estimates: Dict[str, float] = {}
        self.last_rebalance = 0.0
        self.pnl = 0.0

        logger.info(
            f"[{self.strategy_name}] Initialized with "
            f"target_delta={target_delta}, "
            f"basket_size=[{min_basket_size}, {max_basket_size}], "
            f"max_gross_exposure={max_gross_exposure}, "
            f"corr_threshold={corr_threshold}"
        )

    # -------------------------------------------------------------------------
    # Correlation / beta / volatility estimation
    # -------------------------------------------------------------------------
    def update_correlation_matrix(self, price_data: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Update correlation matrix, realized volatility map, and per-market betas.

        price_data: dict[market] -> list[price]
        """
        self.price_history = price_data

        if len(price_data) < 2:
            self.correlation_matrix = None
            self.volatility_map = {}
            self.beta_estimates = {}
            return pd.DataFrame()

        returns_data: Dict[str, np.ndarray] = {}
        for market, prices in price_data.items():
            if len(prices) >= 10:
                returns = np.diff(np.log(prices))
                # cap lookback to 100 points to avoid over-weighting long history
                returns_data[market] = returns[-min(len(returns), 100):]

        if len(returns_data) < 2:
            self.correlation_matrix = None
            self.volatility_map = {}
            self.beta_estimates = {}
            return pd.DataFrame()

        min_len = min(len(r) for r in returns_data.values())
        aligned_returns = {k: v[-min_len:] for k, v in returns_data.items()}

        df = pd.DataFrame(aligned_returns)
        self.correlation_matrix = df.corr()

        # Realized vol per leg (simple, not annualized – your risk engine can choose convention)
        self.volatility_map = {col: float(df[col].std()) for col in df.columns}

        # Beta estimates vs equal-weight index (used for dynamic hedge ratio)
        index_ret = df.mean(axis=1)
        index_var = float(index_ret.var())
        self.beta_estimates = {}
        if index_var > 0:
            for col in df.columns:
                cov = float(np.cov(df[col], index_ret)[0, 1])
                self.beta_estimates[col] = cov / index_var
        else:
            # Fallback: all betas ~1
            self.beta_estimates = {col: 1.0 for col in df.columns}

        logger.info(
            f"[{self.strategy_name}] Updated correlation matrix for {len(df.columns)} markets "
            f"(avg_corr={df.corr().values[np.triu_indices(len(df.columns), 1)].mean():.3f})"
        )

        return self.correlation_matrix

    def _get_vol(self, market: str, market_data: Dict[str, Dict[str, Any]], default: float = 0.5) -> float:
        # market_data may already contain a volatility estimate; fall back to realized
        return float(
            market_data.get(market, {}).get(
                "volatility",
                self.volatility_map.get(market, default),
            )
        )

    def _get_beta(self, market: str, market_data: Dict[str, Dict[str, Any]]) -> float:
        # market_data may already contain a beta; fall back to realized beta vs index
        return float(
            market_data.get(market, {}).get(
                "beta",
                self.beta_estimates.get(market, 1.0),
            )
        )

    # -------------------------------------------------------------------------
    # Metrics / diagnostics
    # -------------------------------------------------------------------------
    def calculate_basket_metrics(self) -> BasketMetrics:
        if not self.basket_positions:
            return BasketMetrics(
                net_delta=0.0,
                gross_exposure=0.0,
                net_funding_apy=0.0,
                hedge_ratio=1.0,
                correlation_avg=0.0,
                is_balanced=True,
            )

        net_delta = sum(p.delta_contribution for p in self.basket_positions.values())
        gross_exposure = sum(abs(p.size * p.weight) for p in self.basket_positions.values())
        net_funding = sum(p.funding_contribution for p in self.basket_positions.values())

        long_exposure = sum(
            p.size * p.weight for p in self.basket_positions.values() if p.side == "long"
        )
        short_exposure = sum(
            abs(p.size * p.weight) for p in self.basket_positions.values() if p.side == "short"
        )

        hedge_ratio = short_exposure / long_exposure if long_exposure > 0 else 1.0

        if self.correlation_matrix is not None and len(self.correlation_matrix) > 0:
            corr_values = self.correlation_matrix.values
            mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)
            correlation_avg = float(np.mean(corr_values[mask])) if mask.any() else 0.0
        else:
            correlation_avg = 0.0

        is_balanced = abs(net_delta - self.target_delta) <= self.delta_tolerance

        return BasketMetrics(
            net_delta=net_delta,
            gross_exposure=gross_exposure,
            net_funding_apy=net_funding,
            hedge_ratio=hedge_ratio,
            correlation_avg=correlation_avg,
            is_balanced=is_balanced,
        )

    # -------------------------------------------------------------------------
    # Portfolio construction helpers (correlation + vol + beta)
    # -------------------------------------------------------------------------
    def _select_low_corr_universe(self, sorted_markets: List[str]) -> List[str]:
        """
        Correlation-aware universe selection: greedily build a set whose
        pairwise correlations are below corr_threshold where possible.
        """
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return sorted_markets[: self.max_basket_size]

        selected: List[str] = []
        for m in sorted_markets:
            if m not in self.correlation_matrix.columns:
                selected.append(m)
                continue

            if not selected:
                selected.append(m)
                continue

            corrs = []
            for s in selected:
                if s in self.correlation_matrix.columns:
                    corrs.append(abs(float(self.correlation_matrix.loc[m, s])))
            max_corr = max(corrs) if corrs else 0.0

            # allow some correlation when we still don't have enough names
            if max_corr <= self.corr_threshold or len(selected) < self.min_basket_size:
                selected.append(m)

            if len(selected) >= self.max_basket_size:
                break

        return selected[: self.max_basket_size]

    def _compute_side_weights_risk_parity(
        self,
        markets: List[str],
        market_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Simple risk-parity style weights: w_i ∝ 1 / vol_i, normalized to sum to 1.

        This is effectively a (very) lightweight portfolio optimization layer that
        tries to equalize per-leg risk contributions.
        """
        if not markets:
            return {}

        vols = {m: max(self._get_vol(m, market_data), 1e-6) for m in markets}
        inv_vols = {m: 1.0 / v for m, v in vols.items()}
        total = sum(inv_vols.values())
        if total <= 0:
            # Fall back to equal weights
            return {m: 1.0 / len(markets) for m in markets}
        return {m: w / total for m, w in inv_vols.items()}

    # -------------------------------------------------------------------------
    # Basket construction (correlation + vol scaler + beta hedge)
    # -------------------------------------------------------------------------
    def construct_basket(
        self,
        market_data: Dict[str, Dict[str, Any]],
        available_capital: float,
    ) -> List[Dict[str, Any]]:
        """
        Correlation-aware, beta-hedged basket construction with volatility-scaled,
        risk-parity style sizing (i.e., a simple portfolio optimization layer).

        Steps:
        1. Rank markets by |funding_apy|
        2. Use correlation matrix (if available) to build a diversified universe
        3. Split into yield-capture shorts (positive funding) and hedge longs (negative funding)
        4. Compute per-side risk-parity weights using realized vol
        5. Allocate capital between sides using beta-based hedge ratio so net-beta ~ target_delta
        """
        actions: List[Dict[str, Any]] = []

        # 1) Rank by |funding_apy|
        ranked = sorted(
            market_data.items(),
            key=lambda x: abs(x[1].get("funding_apy", 0.0)),
            reverse=True,
        )
        # modest oversampling pre-corr filter
        ranked = ranked[: self.max_basket_size * 2]

        if len(ranked) < self.min_basket_size:
            logger.warning(f"[{self.strategy_name}] Insufficient markets for basket construction")
            return []

        market_universe = [m for m, _ in ranked]

        # 2) Correlation-aware universe selection
        diversified_universe = self._select_low_corr_universe(market_universe)

        if len(diversified_universe) < self.min_basket_size:
            diversified_universe = market_universe[: self.min_basket_size]

        # 3) Split into yield-capture (short) vs hedge (long) legs
        long_candidates: List[str] = []
        short_candidates: List[str] = []

        for m in diversified_universe:
            funding_apy = float(market_data.get(m, {}).get("funding_apy", 0.0))
            if funding_apy > 0:
                short_candidates.append(m)
            else:
                long_candidates.append(m)

        if not short_candidates or not long_candidates:
            logger.warning(
                f"[{self.strategy_name}] Unable to form long/short basket "
                f"(shorts={len(short_candidates)}, longs={len(long_candidates)})"
            )
            return []

        # 4) Per-side risk parity weights => volatility scaler
        long_weights = self._compute_side_weights_risk_parity(long_candidates, market_data)
        short_weights = self._compute_side_weights_risk_parity(short_candidates, market_data)

        # 5) Dynamic hedge ratio using beta estimates
        short_betas = {m: abs(self._get_beta(m, market_data)) for m in short_candidates}
        long_betas = {m: abs(self._get_beta(m, market_data)) for m in long_candidates}

        risk_short = sum(short_weights[m] * short_betas[m] for m in short_candidates)
        risk_long = sum(long_weights[m] * long_betas[m] for m in long_candidates)

        if risk_short <= 0 or risk_long <= 0:
            logger.warning(f"[{self.strategy_name}] Invalid beta risk estimates")
            return []

        # Portfolio-level gross exposure budget
        gross_budget = min(self.max_gross_exposure, 2.0) * available_capital

        # Solve for capital_short, capital_long such that:
        #   capital_short * risk_short ≈ capital_long * risk_long (beta-neutral)
        #   capital_short + capital_long = gross_budget
        capital_long = gross_budget / (1.0 + (risk_long / risk_short))
        capital_short = gross_budget - capital_long

        # Guard rails: don't exceed max leverage vs available capital
        if gross_budget > self.max_gross_exposure * available_capital:
            scale = (self.max_gross_exposure * available_capital) / gross_budget
            capital_long *= scale
            capital_short *= scale

        # Turn capital allocations into per-market sizes
        for m in short_candidates:
            size_usd = capital_short * short_weights[m]
            if size_usd <= 0:
                continue
            actions.append(
                {
                    "action_type": "open_perp",
                    "params": {
                        "market": m,
                        "direction": "short",
                        "size_usd": size_usd,
                        "strategy": self.strategy_name,
                        "basket_role": "yield_capture",
                    },
                    "priority": "balanced",
                }
            )

        for m in long_candidates:
            size_usd = capital_long * long_weights[m]
            if size_usd <= 0:
                continue
            actions.append(
                {
                    "action_type": "open_perp",
                    "params": {
                        "market": m,
                        "direction": "long",
                        "size_usd": size_usd,
                        "strategy": self.strategy_name,
                        "basket_role": "delta_hedge",
                    },
                    "priority": "balanced",
                }
            )

        logger.info(
            f"[{self.strategy_name}] Constructed basket: "
            f"{len(short_candidates)} shorts, {len(long_candidates)} longs, "
            f"gross_budget={gross_budget:.2f}"
        )

        return actions

    # -------------------------------------------------------------------------
    # Rebalancing – dynamic delta/beta hedge maintenance
    # -------------------------------------------------------------------------
    def rebalance_basket(self) -> List[Dict[str, Any]]:
        """
        Dynamic rebalancing:
        - Uses current net_delta vs target to decide which side to trim
        - Adjustment size scales with severity of imbalance (beta-weighted delta)
        - Keeps transaction count low by trimming proportionally instead of nuking legs
        """
        metrics = self.calculate_basket_metrics()

        if metrics.is_balanced:
            return []

        actions: List[Dict[str, Any]] = []

        imbalance = metrics.net_delta - self.target_delta
        # Severity scales with multiple of tolerance; capped at 50% trim
        severity = min(0.5, max(0.05, abs(imbalance) / max(self.delta_tolerance, 1e-8) * 0.05))

        if imbalance > 0:
            # Too net-long beta; trim long legs proportionally
            for market, pos in self.basket_positions.items():
                if pos.side == "long" and pos.size > 0:
                    reduction = pos.size * severity
                    actions.append(
                        {
                            "action_type": "reduce_position",
                            "params": {
                                "market": market,
                                "reduction": reduction,
                                "strategy": self.strategy_name,
                                "reason": "delta_rebalance_long_trim",
                            },
                            "priority": "balanced",
                        }
                    )
        else:
            # Too net-short beta; trim shorts
            for market, pos in self.basket_positions.items():
                if pos.side == "short" and pos.size > 0:
                    reduction = pos.size * severity
                    actions.append(
                        {
                            "action_type": "reduce_position",
                            "params": {
                                "market": market,
                                "reduction": reduction,
                                "strategy": self.strategy_name,
                                "reason": "delta_rebalance_short_trim",
                            },
                            "priority": "balanced",
                        }
                    )

        if actions:
            self.last_rebalance = time.time()
            logger.info(
                f"[{self.strategy_name}] Rebalancing triggered: "
                f"net_delta={metrics.net_delta:.4f}, "
                f"target={self.target_delta:.4f}, "
                f"severity={severity:.3f}, "
                f"actions={len(actions)}"
            )

        return actions

    # -------------------------------------------------------------------------
    # Strategy interface
    # -------------------------------------------------------------------------
    def generate_signals(self, market_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_active:
            return []

        # Price history update (if provided) for corr/vol/beta estimation
        price_data = market_snapshot.get("price_history")
        if price_data:
            self.update_correlation_matrix(price_data)

        # When no positions: build initial hedged basket
        if not self.basket_positions:
            funding_rates = market_snapshot.get("funding_rates", {})
            # funding_rates is expected to be: market -> { "apy": float, ... }
            market_data = {
                market: {
                    "funding_apy": data.get("apy", 0.0),
                    # optional, if upstream provides:
                    "volatility": data.get("volatility"),
                    "beta": data.get("beta"),
                }
                for market, data in funding_rates.items()
            }
            return self.construct_basket(market_data, available_capital=market_snapshot.get("capital", 5000.0))

        # Otherwise, manage hedge / rebalance existing basket
        return self.rebalance_basket()

    def apply_simulated_trade(self, action: Dict[str, Any], result: Dict[str, Any]):
        """
        Update internal state after a simulated trade is executed by the backtester.
        `result` is expected to contain at least: success: bool, size: float, pnl: float (for closes).
        """
        market = action.get("params", {}).get("market", "")
        action_type = action.get("action_type")

        if action_type == "open_perp" and result.get("success"):
            direction = action["params"].get("direction", "long")
            size = float(result.get("size", 0.0))

            # Use our beta/volatility estimates to track delta in a portfolio sense
            beta = self.beta_estimates.get(market, 1.0)
            volatility = self.volatility_map.get(market, 0.0)
            sign = 1.0 if direction == "long" else -1.0
            delta_contribution = size * beta * sign

            self.basket_positions[market] = BasketPosition(
                market=market,
                side=direction,
                size=size,
                weight=1.0,  # logical weight; portfolio weights are derived on demand
                delta_contribution=delta_contribution,
                funding_contribution=0.0,
                beta=beta,
                volatility=volatility,
            )

        elif action_type in {"close_perp", "reduce_position"} and result.get("success"):
            if market in self.basket_positions:
                self.pnl += float(result.get("pnl", 0.0))
                if action_type == "close_perp":
                    del self.basket_positions[market]
                else:
                    # For reduce_position we shrink the stored size / delta
                    reduction = float(action.get("params", {}).get("reduction", 0.0))
                    pos = self.basket_positions[market]
                    if reduction > 0 and reduction < pos.size:
                        scale = (pos.size - reduction) / pos.size
                        pos.size *= scale
                        pos.delta_contribution *= scale
                        self.basket_positions[market] = pos
                    else:
                        # Treat as full close if reduction >= size
                        del self.basket_positions[market]

    def get_status(self) -> Dict[str, Any]:
        metrics = self.calculate_basket_metrics()

        return {
            "strategy": self.strategy_name,
            "is_active": self.is_active,
            "positions": {
                market: {
                    "side": pos.side,
                    "size": pos.size,
                    "weight": pos.weight,
                    "delta": pos.delta_contribution,
                    "beta": pos.beta,
                    "volatility": pos.volatility,
                }
                for market, pos in self.basket_positions.items()
            },
            "basket_metrics": {
                "net_delta": metrics.net_delta,
                "gross_exposure": metrics.gross_exposure,
                "net_funding_apy": metrics.net_funding_apy,
                "hedge_ratio": metrics.hedge_ratio,
                "correlation_avg": metrics.correlation_avg,
                "is_balanced": metrics.is_balanced,
            },
            "total_pnl": self.pnl,
            "last_rebalance": self.last_rebalance,
        }
