"""
Agent Bridge Module - Thin wrapper around orchestrator functionality for AI agents.

Provides a SAFE subset of orchestrator functionality as "agent tools":
- Read-only: portfolio state, metrics, prices/basis
- Safe actions (SIMULATION ONLY): run_simulation, update_portfolio_config

IMPORTANT:
- All actions respect existing risk limits and config thresholds
- NEVER bypasses vault_manager / risk_limits
- Does NOT import Flask app or create routes directly
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ToolType(Enum):
    READ_ONLY = "read-only"
    SAFE_ACTION = "safe-action"


@dataclass
class AgentTool:
    name: str
    description: str
    tool_type: ToolType
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_simulation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.tool_type.value,
            "parameters": self.parameters,
            "requires_simulation": self.requires_simulation
        }


class AgentBridge:
    """
    Bridge class that exposes orchestrator functionality as agent tools.
    
    This class does NOT import Flask or create routes. It only provides
    functions/classes that app.py can call to service agent endpoints.
    """
    
    def __init__(self, 
                 data_fetcher=None, 
                 paper_account=None, 
                 vault_manager=None,
                 simulator=None,
                 config=None):
        self.data_fetcher = data_fetcher
        self.paper_account = paper_account
        self.vault_manager = vault_manager
        self.simulator = simulator
        self.config = config
        self._tools: Dict[str, AgentTool] = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all available agent tools."""
        
        self._tools["get_portfolio_state"] = AgentTool(
            name="get_portfolio_state",
            description="Get current portfolio state including NAV, positions, PnL, and allocation breakdown",
            tool_type=ToolType.READ_ONLY,
            handler=self._get_portfolio_state,
            parameters={}
        )
        
        self._tools["get_metrics_snapshot"] = AgentTool(
            name="get_metrics_snapshot",
            description="Get current performance metrics including Sharpe ratio, Sortino ratio, volatility, and drawdown",
            tool_type=ToolType.READ_ONLY,
            handler=self._get_metrics_snapshot,
            parameters={}
        )
        
        self._tools["get_prices_and_basis_map"] = AgentTool(
            name="get_prices_and_basis_map",
            description="Get current prices from all venues and basis spread map across perp venues",
            tool_type=ToolType.READ_ONLY,
            handler=self._get_prices_and_basis_map,
            parameters={
                "symbols": {
                    "type": "array",
                    "description": "List of asset symbols to fetch (optional, defaults to enabled assets)",
                    "required": False
                }
            }
        )
        
        self._tools["get_funding_rates"] = AgentTool(
            name="get_funding_rates",
            description="Get current funding rates across Drift and Hyperliquid venues",
            tool_type=ToolType.READ_ONLY,
            handler=self._get_funding_rates,
            parameters={}
        )
        
        self._tools["get_oracle_health"] = AgentTool(
            name="get_oracle_health",
            description="Get Pyth oracle health status including deviation from composite prices",
            tool_type=ToolType.READ_ONLY,
            handler=self._get_oracle_health,
            parameters={}
        )
        
        self._tools["run_simulation"] = AgentTool(
            name="run_simulation",
            description="Run a Monte Carlo simulation with specified parameters (SIMULATION MODE ONLY)",
            tool_type=ToolType.SAFE_ACTION,
            handler=self._run_simulation,
            parameters={
                "steps": {
                    "type": "integer",
                    "description": "Number of simulation steps (10-500)",
                    "required": False,
                    "default": 50
                },
                "capital": {
                    "type": "number",
                    "description": "Starting capital in USD",
                    "required": False,
                    "default": 10000
                }
            },
            requires_simulation=True
        )
        
        self._tools["update_portfolio_config_simulation"] = AgentTool(
            name="update_portfolio_config_simulation",
            description="Update portfolio configuration in simulation mode (strategy weights, risk limits)",
            tool_type=ToolType.SAFE_ACTION,
            handler=self._update_portfolio_config_simulation,
            parameters={
                "strategy_weights": {
                    "type": "object",
                    "description": "Strategy allocation weights (basis, funding, cash)",
                    "required": False
                },
                "risk_limits": {
                    "type": "object", 
                    "description": "Risk limit overrides (max_position_size_usd, max_leverage, etc)",
                    "required": False
                }
            },
            requires_simulation=True
        )
        
        self._tools["run_scenario"] = AgentTool(
            name="run_scenario",
            description="Run a stress test scenario with price shocks (SIMULATION MODE ONLY)",
            tool_type=ToolType.SAFE_ACTION,
            handler=self._run_scenario,
            parameters={
                "sol_shock": {
                    "type": "number",
                    "description": "SOL price shock as decimal (-0.5 to 0.5)",
                    "required": False,
                    "default": 0
                },
                "btc_shock": {
                    "type": "number",
                    "description": "BTC price shock as decimal (-0.5 to 0.5)",
                    "required": False,
                    "default": 0
                }
            },
            requires_simulation=True
        )
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of all available agent tools."""
        return [tool.to_dict() for tool in self._tools.values()]
    
    def get_tool(self, name: str) -> Optional[AgentTool]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def execute_tool(self, name: str, params: Optional[Dict[str, Any]] = None, 
                     is_simulation: bool = True,
                     agent_kit_live_trading: bool = False) -> Dict[str, Any]:
        """
        Execute an agent tool by name with given parameters.
        
        Args:
            name: Tool name to execute
            params: Parameters to pass to the tool handler
            is_simulation: Whether we're in simulation mode
            agent_kit_live_trading: Whether AGENT_KIT_LIVE_TRADING is enabled
            
        Returns:
            Tool execution result as dict
        """
        tool = self._tools.get(name)
        if not tool:
            return {
                "success": False,
                "error": f"Unknown tool: {name}",
                "available_tools": list(self._tools.keys())
            }
        
        if tool.requires_simulation:
            if not is_simulation:
                return {
                    "success": False,
                    "error": f"Tool '{name}' requires simulation mode. Set MODE=simulation to use this tool.",
                    "tool_type": tool.tool_type.value
                }
            if tool.tool_type == ToolType.SAFE_ACTION and not is_simulation and not agent_kit_live_trading:
                return {
                    "success": False,
                    "error": f"Tool '{name}' requires either simulation mode OR AGENT_KIT_LIVE_TRADING=true.",
                    "tool_type": tool.tool_type.value
                }
        
        try:
            result = tool.handler(params or {}, is_simulation=is_simulation)
            return {
                "success": True,
                "tool": name,
                "tool_type": tool.tool_type.value,
                "result": result
            }
        except Exception as e:
            logger.error(f"Agent tool '{name}' execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": name
            }
    
    def _get_portfolio_state(self, params: Dict, **kwargs) -> Dict[str, Any]:
        """Get current portfolio state."""
        if not self.paper_account:
            return {"error": "Paper account not available"}
        
        state = self.paper_account.get_state()
        positions = self.paper_account.get_positions()
        
        return {
            "nav": state.get("nav", 0),
            "pnl": state.get("pnl", 0),
            "pnl_pct": state.get("pnl_pct", 0),
            "position_count": len(positions),
            "positions": positions,
            "strategy_allocations": state.get("actual_strategy_weights", {}),
            "target_allocations": state.get("target_strategy_weights", {})
        }
    
    def _get_metrics_snapshot(self, params: Dict, **kwargs) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.paper_account:
            return {"error": "Paper account not available"}
        
        metrics = self.paper_account.get_metrics()
        return {
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0),
            "volatility": metrics.get("volatility", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "total_trades": metrics.get("total_trades", 0),
            "profitable_trades": metrics.get("profitable_trades", 0)
        }
    
    def _get_prices_and_basis_map(self, params: Dict, **kwargs) -> Dict[str, Any]:
        """Get current prices and basis spread map."""
        if not self.data_fetcher:
            return {"error": "Data fetcher not available"}
        
        symbols = params.get("symbols")
        
        try:
            prices = self.data_fetcher.get_aggregated_prices()
            
            basis_map = {}
            if hasattr(self.data_fetcher, 'get_basis_map'):
                basis_map = self.data_fetcher.get_basis_map(symbols)
            
            return {
                "prices": prices,
                "basis_map": basis_map,
                "timestamp": self.data_fetcher.last_update_time
            }
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {"error": str(e)}
    
    def _get_funding_rates(self, params: Dict, **kwargs) -> Dict[str, Any]:
        """Get current funding rates."""
        if not self.data_fetcher:
            return {"error": "Data fetcher not available"}
        
        try:
            funding = {}
            if hasattr(self.data_fetcher, 'get_drift_funding_rates'):
                funding["drift"] = self.data_fetcher.get_drift_funding_rates()
            if hasattr(self.data_fetcher, 'get_hl_funding_rates'):
                funding["hyperliquid"] = self.data_fetcher.get_hl_funding_rates()
            
            return {"funding_rates": funding}
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return {"error": str(e)}
    
    def _get_oracle_health(self, params: Dict, **kwargs) -> Dict[str, Any]:
        """Get Pyth oracle health status."""
        if not self.data_fetcher:
            return {"error": "Data fetcher not available"}
        
        try:
            if hasattr(self.data_fetcher, 'get_pyth_health'):
                return self.data_fetcher.get_pyth_health()
            return {"error": "Pyth oracle not available"}
        except Exception as e:
            logger.error(f"Error fetching oracle health: {e}")
            return {"error": str(e)}
    
    def _run_simulation(self, params: Dict, is_simulation: bool = True, **kwargs) -> Dict[str, Any]:
        """Run Monte Carlo simulation. REQUIRES simulation mode."""
        if not is_simulation:
            return {"error": "run_simulation requires simulation mode (MODE=simulation)"}
        
        if not self.simulator:
            return {"error": "Simulator not available"}
        
        steps = min(max(int(params.get("steps", 50)), 10), 500)
        capital = float(params.get("capital", 10000))
        
        try:
            result = self.simulator.run(steps=steps, initial_capital=capital)
            return {
                "steps": steps,
                "initial_capital": capital,
                "final_nav": result.get("final_nav", capital),
                "total_return": result.get("total_return", 0),
                "max_drawdown": result.get("max_drawdown", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "trades": result.get("trade_count", 0)
            }
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {"error": str(e)}
    
    def _update_portfolio_config_simulation(self, params: Dict, is_simulation: bool = True, **kwargs) -> Dict[str, Any]:
        """Update portfolio configuration. REQUIRES simulation mode."""
        if not is_simulation:
            return {"error": "update_portfolio_config_simulation requires simulation mode (MODE=simulation)"}
        
        if not self.vault_manager:
            return {"error": "Vault manager not available"}
        
        try:
            strategy_weights = params.get("strategy_weights", {})
            risk_limits = params.get("risk_limits", {})
            
            if strategy_weights:
                total = sum(strategy_weights.values())
                if abs(total - 1.0) > 0.01:
                    return {"error": f"Strategy weights must sum to 1.0, got {total}"}
            
            if hasattr(self.vault_manager, 'update_config'):
                self.vault_manager.update_config(
                    strategy_weights=strategy_weights,
                    risk_limits=risk_limits
                )
            
            return {
                "updated": True,
                "strategy_weights": strategy_weights,
                "risk_limits": risk_limits
            }
        except Exception as e:
            logger.error(f"Config update error: {e}")
            return {"error": str(e)}
    
    def _run_scenario(self, params: Dict, is_simulation: bool = True, **kwargs) -> Dict[str, Any]:
        """Run stress test scenario. REQUIRES simulation mode."""
        if not is_simulation:
            return {"error": "run_scenario requires simulation mode (MODE=simulation)"}
        
        if not self.paper_account:
            return {"error": "Paper account not available"}
        
        sol_shock = float(params.get("sol_shock", 0))
        btc_shock = float(params.get("btc_shock", 0))
        
        sol_shock = max(min(sol_shock, 0.5), -0.5)
        btc_shock = max(min(btc_shock, 0.5), -0.5)
        
        try:
            current_nav = self.paper_account.get_state().get("nav", 10000)
            
            sol_impact = current_nav * 0.4 * sol_shock
            btc_impact = current_nav * 0.3 * btc_shock
            
            stressed_nav = current_nav + sol_impact + btc_impact
            nav_change = stressed_nav - current_nav
            nav_change_pct = (nav_change / current_nav) * 100 if current_nav > 0 else 0
            
            warnings = []
            if nav_change_pct < -10:
                warnings.append("Scenario exceeds 10% drawdown threshold")
            if nav_change_pct < -20:
                warnings.append("CRITICAL: Scenario exceeds 20% drawdown limit")
            
            return {
                "sol_shock_pct": sol_shock * 100,
                "btc_shock_pct": btc_shock * 100,
                "current_nav": current_nav,
                "stressed_nav": stressed_nav,
                "nav_change": nav_change,
                "nav_change_pct": nav_change_pct,
                "warnings": warnings
            }
        except Exception as e:
            logger.error(f"Scenario error: {e}")
            return {"error": str(e)}


def run_sample_agent_flow(bridge: AgentBridge) -> Dict[str, Any]:
    """
    Demo-only sample flow that demonstrates agent capabilities.
    
    This flow:
    1. Reads current prices
    2. Reads portfolio state
    3. Adjusts portfolio config in simulation
    4. Runs a small scenario test
    
    ONLY runs in simulation mode.
    """
    results = {
        "flow": "sample_agent_demo",
        "steps": []
    }
    
    prices_result = bridge.execute_tool("get_prices_and_basis_map", {})
    results["steps"].append({
        "step": 1,
        "action": "get_prices_and_basis_map",
        "success": prices_result.get("success", False)
    })
    
    portfolio_result = bridge.execute_tool("get_portfolio_state", {})
    results["steps"].append({
        "step": 2,
        "action": "get_portfolio_state",
        "success": portfolio_result.get("success", False),
        "nav": portfolio_result.get("result", {}).get("nav")
    })
    
    config_result = bridge.execute_tool("update_portfolio_config_simulation", {
        "strategy_weights": {"basis": 0.4, "funding": 0.4, "cash": 0.2}
    })
    results["steps"].append({
        "step": 3,
        "action": "update_portfolio_config_simulation",
        "success": config_result.get("success", False)
    })
    
    scenario_result = bridge.execute_tool("run_scenario", {
        "sol_shock": -0.1,
        "btc_shock": -0.05
    })
    results["steps"].append({
        "step": 4,
        "action": "run_scenario",
        "success": scenario_result.get("success", False),
        "stressed_nav": scenario_result.get("result", {}).get("stressed_nav"),
        "warnings": scenario_result.get("result", {}).get("warnings", [])
    })
    
    results["success"] = all(step.get("success", False) for step in results["steps"])
    return results
