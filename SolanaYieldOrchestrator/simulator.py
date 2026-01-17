#!/usr/bin/env python3
import time
import argparse
from typing import Dict, Any, List
from datetime import datetime
import json

import pandas as pd
import numpy as np

from config import Config
from data.data_fetcher import DataFetcher, MarketSnapshot, PriceData, FundingData, BalanceData
from data.analytics import Analytics
from vault.vault_manager import VaultManager
from utils.logging_utils import get_logger, SimulationLogger

logger = get_logger(__name__)

class Simulator:
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.vault_manager = VaultManager(initial_capital)
        self.analytics = Analytics()
        self.sim_logger = SimulationLogger()
        
        self._nav_history: List[Dict[str, Any]] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._snapshot_history: List[MarketSnapshot] = []
        
        self._current_step = 0
        self._total_steps = 0
    
    def generate_mock_snapshot(self, step: int) -> MarketSnapshot:
        base_sol_price = 100.0
        price_variation = np.sin(step * 0.1) * 5 + np.random.normal(0, 2)
        sol_price = base_sol_price + price_variation
        
        spot_prices = {
            "SOL": PriceData(
                symbol="SOL",
                price=sol_price,
                source="simulation",
                change_24h=np.random.uniform(-5, 5)
            ),
            "USDC": PriceData(
                symbol="USDC",
                price=1.0,
                source="simulation"
            ),
            "BTC": PriceData(
                symbol="BTC",
                price=43000 + np.random.normal(0, 500),
                source="simulation"
            ),
            "ETH": PriceData(
                symbol="ETH",
                price=2200 + np.random.normal(0, 50),
                source="simulation"
            )
        }
        
        basis_offset = np.random.uniform(-0.005, 0.01)
        perp_prices = {
            "SOL-PERP": sol_price * (1 + basis_offset),
            "BTC-PERP": 43000 * (1 + np.random.uniform(-0.003, 0.005)),
            "ETH-PERP": 2200 * (1 + np.random.uniform(-0.003, 0.005)),
            "APT-PERP": 8.5 * (1 + np.random.uniform(-0.005, 0.008)),
            "ARB-PERP": 1.2 * (1 + np.random.uniform(-0.004, 0.006))
        }
        
        funding_rates = {}
        for market in perp_prices:
            rate = np.random.uniform(-0.0002, 0.0003)
            apy = rate * 24 * 365 * 100
            funding_rates[market] = FundingData(
                market=market,
                funding_rate=rate,
                funding_apy=apy
            )
        
        balances = {
            "USDC": BalanceData(
                token="USDC",
                amount=self.vault_manager.state.cash_balance,
                value_usd=self.vault_manager.state.cash_balance
            ),
            "SOL": BalanceData(
                token="SOL",
                amount=0,
                value_usd=0
            )
        }
        
        return MarketSnapshot(
            timestamp=time.time(),
            spot_prices=spot_prices,
            perp_prices=perp_prices,
            funding_rates=funding_rates,
            balances=balances,
            metadata={"step": step, "mode": "simulation"}
        )
    
    def run_step(self, snapshot: MarketSnapshot = None) -> Dict[str, Any]:
        if snapshot is None:
            snapshot = self.generate_mock_snapshot(self._current_step)
        
        self._snapshot_history.append(snapshot)
        
        actions = self.vault_manager.update(snapshot)
        
        results = self.vault_manager.execute_pending_actions()
        
        for result in results:
            self._trade_history.append(result)
            self.sim_logger.events.append({
                "type": "trade",
                "step": self._current_step,
                **result
            })
        
        status = self.vault_manager.get_status()
        
        self._nav_history.append({
            "step": self._current_step,
            "timestamp": time.time(),
            "nav": status["vault_state"]["total_nav"],
            "pnl": status["portfolio_metrics"]["total_pnl"],
            "pnl_pct": status["portfolio_metrics"]["total_pnl_pct"]
        })
        
        self._current_step += 1
        
        return {
            "step": self._current_step,
            "actions": len(actions),
            "nav": status["vault_state"]["total_nav"],
            "status": status
        }
    
    def run_simulation(self, num_steps: int = 100, step_interval_seconds: float = 0.0) -> Dict[str, Any]:
        logger.info(f"Starting simulation: {num_steps} steps, initial capital: ${self.initial_capital}")
        
        self._total_steps = num_steps
        start_time = time.time()
        
        for step in range(num_steps):
            self.run_step()
            
            if step_interval_seconds > 0:
                time.sleep(step_interval_seconds)
            
            if (step + 1) % 10 == 0:
                current_nav = self._nav_history[-1]["nav"]
                pnl_pct = self._nav_history[-1]["pnl_pct"]
                logger.info(f"Step {step + 1}/{num_steps}: NAV=${current_nav:.2f}, PnL={pnl_pct:.2f}%")
        
        elapsed = time.time() - start_time
        
        results = self.get_simulation_results()
        results["elapsed_time"] = elapsed
        results["steps_per_second"] = num_steps / elapsed if elapsed > 0 else 0
        
        logger.info(f"Simulation complete: {num_steps} steps in {elapsed:.2f}s")
        logger.info(f"Final NAV: ${results['final_nav']:.2f}, Total Return: {results['total_return_pct']:.2f}%")
        
        return results
    
    def get_simulation_results(self) -> Dict[str, Any]:
        if not self._nav_history:
            return {
                "error": "No simulation data available",
                "steps": 0
            }
        
        navs = [h["nav"] for h in self._nav_history]
        
        final_nav = navs[-1]
        total_return = final_nav - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        peak = navs[0]
        max_drawdown = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        if len(navs) > 1:
            returns = np.diff(navs) / navs[:-1]
            volatility = np.std(returns) * np.sqrt(365) * 100
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
        else:
            volatility = 0
            sharpe = 0
        
        strategy_status = self.vault_manager.get_status()["strategies"]
        
        strategy_pnl = {}
        for name, status in strategy_status.items():
            realized = status.get("realized_pnl", status.get("total_funding_earned", 0))
            unrealized = status.get("unrealized_pnl", status.get("accumulated_funding", 0))
            strategy_pnl[name] = {
                "realized": realized,
                "unrealized": unrealized,
                "total": realized + unrealized
            }
        
        return {
            "steps": self._current_step,
            "initial_capital": self.initial_capital,
            "final_nav": final_nav,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe,
            "total_trades": len(self._trade_history),
            "strategy_pnl": strategy_pnl,
            "nav_history": self._nav_history[-100:],
            "trade_history": self._trade_history[-50:]
        }
    
    def get_nav_dataframe(self) -> pd.DataFrame:
        if not self._nav_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._nav_history)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df
    
    def export_results(self, filepath: str = "simulation_results.json"):
        results = self.get_simulation_results()
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")
    
    def reset(self):
        self.vault_manager.reset(self.initial_capital)
        self._nav_history.clear()
        self._trade_history.clear()
        self._snapshot_history.clear()
        self._current_step = 0
        self._total_steps = 0
        self.sim_logger.clear()


def main():
    parser = argparse.ArgumentParser(description="Solana Yield Orchestrator Simulator")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in USDC")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--output", type=str, default="simulation_results.json", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    simulator = Simulator(initial_capital=args.capital)
    results = simulator.run_simulation(num_steps=args.steps)
    simulator.export_results(args.output)
    
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Steps:           {results['steps']}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final NAV:       ${results['final_nav']:,.2f}")
    print(f"Total Return:    {results['total_return_pct']:.2f}%")
    print(f"Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
    print(f"Volatility:      {results['volatility_pct']:.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Total Trades:    {results['total_trades']}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    main()
