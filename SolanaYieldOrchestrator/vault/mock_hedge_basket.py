import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import threading

from config import Config
from strategies import Strategy, Action
from data.analytics import Analytics
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BasketAsset:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: float
    hedge_ratio: float = 1.0
    beta: float = 1.0
    volatility: float = 0.0


class CorrelationMatrix:
    """Compute and maintain correlation matrix for basket assets"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: Dict[str, deque] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.symbols: List[str] = []
        self._lock = threading.Lock()
    
    def add_price(self, symbol: str, price: float):
        """Add price data for a symbol"""
        with self._lock:
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.window_size)
            self.price_history[symbol].append(price)
    
    def compute_correlation(self) -> np.ndarray:
        """Compute correlation matrix from price history"""
        with self._lock:
            symbols = list(self.price_history.keys())
            if len(symbols) < 2:
                return np.eye(len(symbols))
            
            # Get prices and compute returns
            prices = []
            for symbol in symbols:
                if len(self.price_history[symbol]) > 1:
                    prices.append(list(self.price_history[symbol]))
                else:
                    return np.eye(len(symbols))
            
            # Convert to numpy array
            price_array = np.array(prices)
            
            # Compute log returns
            returns = np.diff(np.log(price_array), axis=1)
            
            # Compute correlation
            correlation = np.corrcoef(returns)
            
            self.symbols = symbols
            self.correlation_matrix = correlation
            return correlation
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        if self.correlation_matrix is None:
            self.compute_correlation()
        
        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            return 0.0
        
        idx1 = self.symbols.index(symbol1)
        idx2 = self.symbols.index(symbol2)
        
        if self.correlation_matrix is not None:
            return float(self.correlation_matrix[idx1, idx2])
        return 0.0


class BetaEstimator:
    """Estimate beta relative to market (typically SOL)"""
    
    def __init__(self, market_symbol: str = "SOL", window_size: int = 50):
        self.market_symbol = market_symbol
        self.window_size = window_size
        self.market_returns: deque = deque(maxlen=window_size)
        self.asset_returns: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def add_return(self, symbol: str, market_return: float, asset_return: float):
        """Add return data for beta calculation"""
        with self._lock:
            self.market_returns.append(market_return)
            if symbol not in self.asset_returns:
                self.asset_returns[symbol] = deque(maxlen=self.window_size)
            self.asset_returns[symbol].append(asset_return)
    
    def estimate_beta(self, symbol: str) -> float:
        """Estimate beta for a symbol"""
        with self._lock:
            if symbol not in self.asset_returns or len(self.market_returns) < 2:
                return 1.0
            
            market_ret = np.array(list(self.market_returns))
            asset_ret = np.array(list(self.asset_returns[symbol]))
            
            if len(market_ret) != len(asset_ret):
                return 1.0
            
            # Covariance between asset and market
            covariance = np.cov(asset_ret, market_ret)[0, 1]
            
            # Variance of market
            market_variance = np.var(market_ret)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return float(np.clip(beta, 0.1, 5.0))


class VolatilityScaler:
    """Scale positions based on volatility"""
    
    def __init__(self, window_size: int = 30, target_vol: float = 0.15):
        self.window_size = window_size
        self.target_vol = target_vol
        self.returns_history: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def add_return(self, symbol: str, return_pct: float):
        """Add return data for volatility calculation"""
        with self._lock:
            if symbol not in self.returns_history:
                self.returns_history[symbol] = deque(maxlen=self.window_size)
            self.returns_history[symbol].append(return_pct)
    
    def estimate_volatility(self, symbol: str) -> float:
        """Estimate volatility (annualized)"""
        with self._lock:
            if symbol not in self.returns_history or len(self.returns_history[symbol]) < 2:
                return 0.15
            
            returns = np.array(list(self.returns_history[symbol]))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            return float(np.clip(volatility, 0.05, 2.0))
    
    def get_position_size_scaler(self, current_vol: float) -> float:
        """Get position size scaler based on volatility normalization"""
        if current_vol == 0:
            return 1.0
        
        scaler = self.target_vol / current_vol
        return float(np.clip(scaler, 0.5, 2.0))


class PortfolioOptimizer:
    """Portfolio optimization using mean-variance or risk parity"""
    
    @staticmethod
    def optimize_mean_variance(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_return: float = None
    ) -> np.ndarray:
        """Optimize portfolio using mean-variance optimization
        
        Returns optimal weights that maximize Sharpe ratio
        """
        n_assets = len(expected_returns)
        
        if n_assets < 2:
            return np.array([1.0] + [0.0] * (n_assets - 1))
        
        # Ensure covariance matrix is valid
        if cov_matrix.shape != (n_assets, n_assets):
            cov_matrix = np.eye(n_assets) * 0.1
        
        try:
            # Add regularization to covariance matrix
            cov_matrix = cov_matrix + np.eye(n_assets) * 0.001
            
            # Compute inverse of covariance matrix
            inv_cov = np.linalg.inv(cov_matrix)
            
            # Ones vector
            ones = np.ones(n_assets)
            
            # Optimal weights proportional to inverse covariance
            weights = inv_cov @ ones
            weights = weights / np.sum(weights)
            
            # Clip weights to reasonable bounds
            weights = np.clip(weights, 0, 0.5)
            weights = weights / np.sum(weights)
            
            return weights
        except np.linalg.LinAlgError:
            # If inversion fails, use equal weight
            return np.ones(n_assets) / n_assets
    
    @staticmethod
    def optimize_risk_parity(volatilities: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize portfolio using risk parity (inverse volatility weighting)
        
        Returns weights inversely proportional to volatility
        """
        if len(volatilities) == 0:
            return np.array([])
        
        # Avoid division by zero
        safe_vols = np.clip(volatilities, 1e-4, None)
        
        # Inverse volatility weights
        weights = 1.0 / safe_vols
        weights = weights / np.sum(weights)
        
        return weights


class HedgeBasket(Strategy):
    """Hedge Basket Strategy: correlation-based construction with dynamic hedging"""
    
    def __init__(self, target_alloc: float = None, optimization_method: str = "risk_parity"):
        self._target_allocation = target_alloc or Config.get_allocations().get("hedge_basket", 0.15)
        self.optimization_method = optimization_method  # "mean_variance" or "risk_parity"
        
        self.analytics = Analytics()
        self.basket_assets: Dict[str, BasketAsset] = {}
        
        # Components
        self.correlation_matrix = CorrelationMatrix(window_size=100)
        self.beta_estimator = BetaEstimator(market_symbol="SOL", window_size=50)
        self.volatility_scaler = VolatilityScaler(window_size=30, target_vol=0.15)
        self.portfolio_optimizer = PortfolioOptimizer()
        
        self._current_snapshot = None
        self._last_update = 0
        self._total_pnl = 0.0
        self._rebalance_interval = 3600  # 1 hour
        self._last_rebalance = 0
        self._trade_count = 0
    
    def name(self) -> str:
        return "hedge_basket"
    
    def target_allocation(self) -> float:
        return self._target_allocation
    
    def update_state(self, market_snapshot) -> None:
        """Update internal state with new market data"""
        self._current_snapshot = market_snapshot
        self._last_update = time.time()
        
        # Update price histories for correlation
        self._update_price_histories()
        
        # Update position PnL
        self._update_position_pnl()
        
        # Check if rebalancing needed
        if time.time() - self._last_rebalance > self._rebalance_interval:
            self._rebalance_basket()
    
    def _update_price_histories(self):
        """Update price histories from market snapshot"""
        if not self._current_snapshot:
            return
        
        # Add spot prices
        if hasattr(self._current_snapshot, 'spot_prices'):
            for symbol, price_data in self._current_snapshot.spot_prices.items():
                price = price_data.price if hasattr(price_data, 'price') else price_data
                if price > 0:
                    self.correlation_matrix.add_price(symbol, price)
                    
                    # Update volatility scaler
                    if symbol in self.basket_assets:
                        prev_price = self.basket_assets[symbol].entry_price
                        ret = (price - prev_price) / prev_price if prev_price > 0 else 0
                        self.volatility_scaler.add_return(symbol, ret)
    
    def _update_position_pnl(self):
        """Update PnL for all positions"""
        if not self._current_snapshot:
            return
        
        total_pnl = 0.0
        for symbol, asset in self.basket_assets.items():
            # Get current price
            if hasattr(self._current_snapshot, 'spot_prices'):
                current_data = self._current_snapshot.spot_prices.get(symbol)
                if current_data:
                    current_price = current_data.price if hasattr(current_data, 'price') else current_data
                    pnl = (current_price - asset.entry_price) * asset.quantity
                    total_pnl += pnl
        
        self._total_pnl = total_pnl
    
    def _rebalance_basket(self):
        """Rebalance basket based on correlation and optimization"""
        if len(self.basket_assets) < 2:
            return
        
        # Compute correlation matrix
        corr_matrix = self.correlation_matrix.compute_correlation()
        
        # Estimate betas
        symbols = list(self.basket_assets.keys())
        for symbol in symbols:
            beta = self.beta_estimator.estimate_beta(symbol)
            self.basket_assets[symbol].beta = beta
        
        # Estimate volatilities
        volatilities = []
        for symbol in symbols:
            vol = self.volatility_scaler.estimate_volatility(symbol)
            self.basket_assets[symbol].volatility = vol
            volatilities.append(vol)
        
        # Optimize portfolio weights
        volatilities_arr = np.array(volatilities)
        
        if self.optimization_method == "risk_parity":
            optimal_weights = self.portfolio_optimizer.optimize_risk_parity(
                volatilities_arr, corr_matrix
            )
        else:
            expected_returns = np.array([0.1] * len(symbols))
            optimal_weights = self.portfolio_optimizer.optimize_mean_variance(
                expected_returns, corr_matrix
            )
        
        # Update hedge ratios based on optimization
        for i, symbol in enumerate(symbols):
            if i < len(optimal_weights):
                # Dynamic hedge ratio based on beta and volatility
                beta = self.basket_assets[symbol].beta
                vol = self.basket_assets[symbol].volatility
                
                # Hedge ratio = weight / beta (adjust for market exposure)
                base_hedge = optimal_weights[i]
                hedge_ratio = base_hedge / max(beta, 0.1)
                
                # Apply volatility scaling
                vol_scaler = self.volatility_scaler.get_position_size_scaler(vol)
                final_hedge_ratio = hedge_ratio * vol_scaler
                
                self.basket_assets[symbol].hedge_ratio = float(np.clip(final_hedge_ratio, 0.1, 2.0))
        
        self._last_rebalance = time.time()
        logger.info(f"[{self.name()}] Rebalanced basket with {len(symbols)} assets")
    
    def _construct_basket(self, vault_state: Dict[str, Any]) -> bool:
        """Construct initial basket based on correlations"""
        if not self._current_snapshot:
            return False
        
        # Select assets with low correlation (diversification)
        if not hasattr(self._current_snapshot, 'spot_prices'):
            return False
        
        symbols = list(self._current_snapshot.spot_prices.keys())
        symbols = [s for s in symbols if s in ["SOL", "BTC", "ETH", "mSOL"]][:3]
        
        if len(symbols) < 2:
            return False
        
        available_per_asset = vault_state.get("available_capital", 0) / len(symbols)
        
        for symbol in symbols:
            if symbol not in self.basket_assets:
                price_data = self._current_snapshot.spot_prices.get(symbol)
                if price_data:
                    price = price_data.price if hasattr(price_data, 'price') else price_data
                    if price > 0:
                        quantity = available_per_asset / price
                        self.basket_assets[symbol] = BasketAsset(
                            symbol=symbol,
                            quantity=quantity,
                            entry_price=price,
                            entry_time=time.time(),
                            hedge_ratio=1.0,
                            beta=1.0
                        )
        
        return len(self.basket_assets) > 0
    
    def desired_actions(self, vault_state: Dict[str, Any]) -> List[Action]:
        """Generate actions for basket construction and hedging"""
        actions = []
        
        if not self._current_snapshot:
            return actions
        
        available_capital = vault_state.get("available_capital", 0) * self._target_allocation
        
        if available_capital < 100:
            return actions
        
        # Construct basket if needed
        if len(self.basket_assets) == 0:
            if self._construct_basket(vault_state):
                # Create actions for basket construction
                for symbol, asset in self.basket_assets.items():
                    actions.append(Action(
                        action_type="swap",
                        params={
                            "symbol": symbol,
                            "size": asset.quantity,
                            "direction": "buy"
                        },
                        priority="balanced"
                    ))
                logger.info(f"[{self.name()}] Constructed basket with {len(self.basket_assets)} assets")
        
        # Rebalance if needed
        elif time.time() - self._last_rebalance > self._rebalance_interval:
            self._rebalance_basket()
        
        return actions
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        assets_info = {}
        for symbol, asset in self.basket_assets.items():
            assets_info[symbol] = {
                "quantity": asset.quantity,
                "entry_price": asset.entry_price,
                "hedge_ratio": asset.hedge_ratio,
                "beta": asset.beta,
                "volatility": asset.volatility
            }
        
        return {
            "strategy": self.name(),
            "target_allocation": self._target_allocation,
            "basket_size": len(self.basket_assets),
            "assets": assets_info,
            "total_pnl": self._total_pnl,
            "total_pnl_pct": (self._total_pnl / sum(a.entry_price * a.quantity for a in self.basket_assets.values()) * 100) if self.basket_assets else 0,
            "trade_count": self._trade_count,
            "optimization_method": self.optimization_method,
            "last_rebalance": self._last_rebalance
        }
    
    def apply_simulated_trade(self, action: Action, success: bool = True):
        """Apply simulated trade to update internal state"""
        if success and action.action_type == "swap":
            self._trade_count += 1
