import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from config import Config
from strategies import Strategy, Action
from strategies.basis_harvester import BasisHarvester
from strategies.funding_rotator import FundingRotator
from data.analytics import Analytics
from utils.logging_utils import get_logger
from utils.risk_limits import RiskLimits

logger = get_logger(__name__)

_portfolio_config_cache: Dict[str, Any] = {}

@dataclass
class VaultState:
    total_usdc: float = 0.0
    total_nav: float = 0.0
    strategy_allocations: Dict[str, float] = field(default_factory=dict)
    strategy_pnl: Dict[str, float] = field(default_factory=dict)
    cash_balance: float = 0.0
    last_update: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_usdc": self.total_usdc,
            "total_nav": self.total_nav,
            "strategy_allocations": self.strategy_allocations,
            "strategy_pnl": self.strategy_pnl,
            "cash_balance": self.cash_balance,
            "last_update": self.last_update,
            "allocation_pcts": {
                k: (v / self.total_nav * 100) if self.total_nav > 0 else 0
                for k, v in self.strategy_allocations.items()
            }
        }

class VaultManager:
    def __init__(self, initial_capital: float = 10000.0, session_id: str = "default"):
        self.initial_capital = initial_capital
        self.session_id = session_id
        self.state = VaultState(
            total_usdc=initial_capital,
            total_nav=initial_capital,
            cash_balance=initial_capital
        )
        
        self.strategies: Dict[str, Strategy] = {}
        self.analytics = Analytics()
        self.risk_limits = RiskLimits()
        
        self._pending_actions: List[Action] = []
        self._execution_history: List[Dict[str, Any]] = []
        self._cycle_count = 0
        
        self._custom_allocations: Optional[Dict[str, float]] = None
        
        self._init_strategies()
    
    def _init_strategies(self):
        self.strategies["basis_harvester"] = BasisHarvester(
            target_alloc=Config.ALLOCATION_BASIS_HARVESTER,
            entry_threshold_bps=Config.BASIS_ENTRY_THRESHOLD_BPS,
            exit_threshold_bps=Config.BASIS_EXIT_THRESHOLD_BPS
        )
        
        self.strategies["funding_rotator"] = FundingRotator(
            target_alloc=Config.ALLOCATION_FUNDING_ROTATOR,
            top_n_markets=Config.FUNDING_TOP_N_MARKETS,
            min_apy_threshold=Config.FUNDING_MIN_APY_THRESHOLD,
            rotation_epoch_seconds=Config.ROTATION_EPOCH_SECONDS
        )
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def update(self, market_snapshot) -> List[Action]:
        self._cycle_count += 1
        
        for name, strategy in self.strategies.items():
            strategy.update_state(market_snapshot)
        
        self._update_nav(market_snapshot)
        
        self.analytics.record_nav(self.state.total_nav, self.state.strategy_allocations)
        
        all_actions = []
        vault_state_dict = self._get_vault_state_for_strategies()
        
        for name, strategy in self.strategies.items():
            try:
                actions = strategy.desired_actions(vault_state_dict)
                for action in actions:
                    action.params["strategy"] = name
                all_actions.extend(actions)
            except Exception as e:
                logger.error(f"Error getting actions from {name}: {e}")
        
        self._pending_actions = all_actions
        self.state.last_update = time.time()
        
        return all_actions
    
    def _update_nav(self, market_snapshot):
        total_strategy_value = 0.0
        
        for name, strategy in self.strategies.items():
            status = strategy.get_status()
            position_value = status.get("position_value_usd", 0)
            unrealized_pnl = status.get("unrealized_pnl", 0)
            
            strategy_value = position_value + unrealized_pnl
            self.state.strategy_allocations[name] = strategy_value
            
            realized_pnl = status.get("realized_pnl", status.get("total_funding_earned", 0))
            self.state.strategy_pnl[name] = realized_pnl
            
            total_strategy_value += strategy_value
        
        total_realized_pnl = sum(self.state.strategy_pnl.values())
        
        self.state.total_nav = self.state.cash_balance + total_strategy_value + total_realized_pnl
    
    def set_custom_allocations(self, strategy_weights: Dict[str, float]):
        self._custom_allocations = {
            "basis_harvester": strategy_weights.get("basis", 0.5),
            "funding_rotator": strategy_weights.get("funding", 0.3),
            "cash": strategy_weights.get("cash", 0.2),
        }
        logger.info(f"Custom allocations set: {self._custom_allocations}")
    
    def get_target_allocations(self) -> Dict[str, float]:
        if self._custom_allocations:
            return self._custom_allocations
        return Config.get_allocations()
    
    def _get_vault_state_for_strategies(self) -> Dict[str, Any]:
        target_allocations = self.get_target_allocations()
        cash_target = target_allocations.get("cash", 0.20)
        
        reserved_cash = self.state.total_nav * cash_target
        available = max(0, self.state.cash_balance - reserved_cash)
        
        return {
            "total_nav": self.state.total_nav,
            "cash_balance": self.state.cash_balance,
            "available_capital": available,
            "strategy_allocations": self.state.strategy_allocations.copy(),
            "target_allocations": target_allocations
        }
    
    def execute_pending_actions(self, priority_router=None) -> List[Dict[str, Any]]:
        results = []
        
        for action in self._pending_actions:
            result = self._execute_action(action, priority_router)
            results.append(result)
            self._execution_history.append(result)
        
        self._pending_actions.clear()
        
        return results
    
    def _check_oracle_sanity(self, action: Action) -> tuple:
        """
        Check oracle sanity before opening new positions.
        Returns (is_trusted, deviation_bps, reason)
        """
        if not Config.PYTH_ENABLED:
            return True, None, None
        
        if action.action_type == "close_perp":
            return True, None, "close_position_always_allowed"
        
        market = action.params.get("market", "")
        symbol = market.replace("-PERP", "").replace("-SPOT", "") if market else None
        
        if not symbol:
            return True, None, "no_symbol"
        
        try:
            from infra.pyth_client import get_pyth_client
            from data.data_fetcher import DataFetcher
            
            pyth = get_pyth_client()
            
            composite_price = 100.0
            try:
                from trading.paper_account import get_paper_account
                pa = get_paper_account()
                if hasattr(pa, 'spot_positions') and symbol in pa.spot_positions:
                    composite_price = pa.spot_positions[symbol].current_price
            except:
                pass
            
            if not pyth.is_oracle_trusted(symbol, composite_price):
                deviation = pyth.get_pyth_deviation(symbol, composite_price)
                return False, deviation, f"oracle_deviation_exceeded_{symbol}"
            
            return True, None, None
            
        except Exception as e:
            logger.debug(f"Oracle sanity check failed: {e}")
            return True, None, "check_failed"
    
    def _emit_oracle_block_event(self, action: Action, deviation_bps: float, reason: str):
        """Emit an event when a trade is blocked due to oracle deviation."""
        try:
            from infra.redis_client import RedisCache
            import json
            
            event = {
                "type": "oracle_block",
                "action": action.to_dict(),
                "deviation_bps": deviation_bps,
                "reason": reason,
                "max_allowed_bps": Config.MAX_ORACLE_DEVIATION_BPS,
                "timestamp": time.time()
            }
            
            RedisCache.lpush("yield_orch:events", json.dumps(event))
            logger.warning(f"Trade blocked due to oracle deviation: {reason}, deviation={deviation_bps:.1f}bps")
            
        except Exception as e:
            logger.debug(f"Failed to emit oracle block event: {e}")
    
    def _execute_action(self, action: Action, priority_router=None) -> Dict[str, Any]:
        strategy_name = action.params.get("strategy", "unknown")
        
        is_trusted, deviation_bps, reason = self._check_oracle_sanity(action)
        if not is_trusted:
            self._emit_oracle_block_event(action, deviation_bps, reason)
            return {
                "success": False,
                "blocked": True,
                "reason": "oracle_deviation_exceeded",
                "deviation_bps": deviation_bps,
                "max_allowed_bps": Config.MAX_ORACLE_DEVIATION_BPS,
                "action": action.to_dict(),
                "timestamp": time.time()
            }
        
        if Config.is_simulation():
            success = True
            
            strategy = self.strategies.get(strategy_name)
            if strategy and hasattr(strategy, 'apply_simulated_trade'):
                strategy.apply_simulated_trade(action, success)
            
            return {
                "success": True,
                "simulated": True,
                "action": action.to_dict(),
                "timestamp": time.time()
            }
        
        if priority_router:
            from infra.priority_router import Action as RouterAction, ActionType, PriorityProfile
            
            action_type_map = {
                "swap": ActionType.SWAP,
                "open_perp": ActionType.OPEN_PERP,
                "close_perp": ActionType.CLOSE_PERP
            }
            
            priority_map = {
                "cheap": PriorityProfile.CHEAP,
                "balanced": PriorityProfile.BALANCED,
                "fast": PriorityProfile.FAST
            }
            
            router_action = RouterAction(
                action_type=action_type_map.get(action.action_type, ActionType.SWAP),
                params=action.params,
                priority_profile=priority_map.get(action.priority, PriorityProfile.BALANCED)
            )
            
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(priority_router.execute_action(router_action))
                return {
                    "success": result.success,
                    "simulated": result.simulated,
                    "signature": result.signature,
                    "error": result.error,
                    "action": action.to_dict(),
                    "timestamp": time.time()
                }
            finally:
                loop.close()
        
        return {
            "success": False,
            "error": "No priority router available",
            "action": action.to_dict(),
            "timestamp": time.time()
        }
    
    def get_status(self) -> Dict[str, Any]:
        strategy_statuses = {}
        for name, strategy in self.strategies.items():
            strategy_statuses[name] = strategy.get_status()
        
        portfolio_metrics = self.analytics.get_portfolio_metrics(self.state.total_nav)
        
        risk_checks = self.risk_limits.run_all_checks(current_nav=self.state.total_nav)
        risk_summary = self.risk_limits.get_risk_summary(risk_checks)
        
        return {
            "vault_state": self.state.to_dict(),
            "strategies": strategy_statuses,
            "portfolio_metrics": {
                "total_nav": portfolio_metrics.total_nav,
                "total_pnl": portfolio_metrics.total_pnl,
                "total_pnl_pct": portfolio_metrics.total_pnl_pct,
                "daily_pnl": portfolio_metrics.daily_pnl,
                "max_drawdown": portfolio_metrics.max_drawdown,
                "sharpe_ratio": portfolio_metrics.sharpe_ratio,
                "volatility": portfolio_metrics.volatility
            },
            "risk": risk_summary,
            "cycle_count": self._cycle_count,
            "pending_actions": len(self._pending_actions),
            "execution_history_count": len(self._execution_history),
            "mode": Config.MODE
        }
    
    def get_allocations(self) -> Dict[str, Any]:
        target = self.get_target_allocations()
        
        actual = {}
        for name in self.strategies:
            if self.state.total_nav > 0:
                actual[name] = self.state.strategy_allocations.get(name, 0) / self.state.total_nav
            else:
                actual[name] = 0
        
        actual["cash"] = self.state.cash_balance / self.state.total_nav if self.state.total_nav > 0 else 1.0
        
        return {
            "target": target,
            "actual": actual,
            "deviation": {
                k: abs(target.get(k, 0) - actual.get(k, 0)) * 100
                for k in set(target.keys()) | set(actual.keys())
            },
            "custom_allocations_active": self._custom_allocations is not None
        }
    
    def deposit(self, amount: float):
        self.state.cash_balance += amount
        self.state.total_usdc += amount
        self.state.total_nav += amount
        logger.info(f"Deposited ${amount:.2f}, new NAV: ${self.state.total_nav:.2f}")
    
    def withdraw(self, amount: float) -> bool:
        if amount > self.state.cash_balance:
            logger.warning(f"Insufficient cash for withdrawal: ${amount:.2f} requested, ${self.state.cash_balance:.2f} available")
            return False
        
        self.state.cash_balance -= amount
        self.state.total_usdc -= amount
        self.state.total_nav -= amount
        logger.info(f"Withdrew ${amount:.2f}, new NAV: ${self.state.total_nav:.2f}")
        return True
    
    def reset(self, initial_capital: float = None):
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        self.state = VaultState(
            total_usdc=self.initial_capital,
            total_nav=self.initial_capital,
            cash_balance=self.initial_capital
        )
        
        self._pending_actions.clear()
        self._execution_history.clear()
        self._cycle_count = 0
        
        self.analytics.reset()
        self.risk_limits.reset_peak_nav()
        
        self._init_strategies()
        
        logger.info(f"Vault reset with capital: ${self.initial_capital:.2f}")
