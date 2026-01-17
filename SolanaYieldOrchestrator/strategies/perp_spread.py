from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SpreadPosition:
    long_market: str
    short_market: str
    size: float
    entry_spread: float
    current_spread: float
    unrealized_pnl: float
    opened_at: float


class PerpSpreadStrategy:
    def __init__(
        self,
        drift_client=None,
        spread_pairs: Optional[List[tuple]] = None,
        entry_threshold_bps: float = 50,
        exit_threshold_bps: float = 10,
        max_position_usd: float = 5000.0
    ):
        self.drift_client = drift_client
        self.spread_pairs = spread_pairs or [
            ("SOL-PERP", "mSOL-PERP"),
        ]
        self.entry_threshold_bps = entry_threshold_bps
        self.exit_threshold_bps = exit_threshold_bps
        self.max_position_usd = max_position_usd
        
        self.positions: List[SpreadPosition] = []
        self.pnl_history: List[Dict[str, Any]] = []
        self.total_pnl: float = 0.0
    
    def analyze(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        actions = []
        
        for long_market, short_market in self.spread_pairs:
            spread_info = self._calculate_spread(long_market, short_market, market_data)
            
            if spread_info is None:
                continue
            
            spread_bps = spread_info["spread_bps"]
            
            existing_position = self._get_position(long_market, short_market)
            
            if existing_position:
                existing_position.current_spread = spread_bps
                existing_position.unrealized_pnl = self._calculate_position_pnl(existing_position)
                
                if abs(spread_bps) < self.exit_threshold_bps:
                    actions.append({
                        "action_type": "close_spread",
                        "params": {
                            "long_market": long_market,
                            "short_market": short_market,
                            "reason": "spread_convergence",
                            "spread_bps": spread_bps,
                            "strategy": "perp_spread"
                        },
                        "priority": "balanced"
                    })
            else:
                if abs(spread_bps) >= self.entry_threshold_bps:
                    if spread_bps > 0:
                        actions.append({
                            "action_type": "open_spread",
                            "params": {
                                "long_market": short_market,
                                "short_market": long_market,
                                "spread_bps": spread_bps,
                                "size_usd": self.max_position_usd,
                                "strategy": "perp_spread"
                            },
                            "priority": "balanced"
                        })
                    else:
                        actions.append({
                            "action_type": "open_spread",
                            "params": {
                                "long_market": long_market,
                                "short_market": short_market,
                                "spread_bps": abs(spread_bps),
                                "size_usd": self.max_position_usd,
                                "strategy": "perp_spread"
                            },
                            "priority": "balanced"
                        })
        
        return actions
    
    def _calculate_spread(
        self,
        market_a: str,
        market_b: str,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        price_a = None
        price_b = None
        funding_a = 0.0
        funding_b = 0.0
        
        if self.drift_client:
            perp_markets = self.drift_client.get_perp_markets()
            for market in perp_markets:
                if market.symbol == market_a:
                    price_a = market.mark_price
                    funding_a = market.funding_rate
                elif market.symbol == market_b:
                    price_b = market.mark_price
                    funding_b = market.funding_rate
        
        if price_a is None or price_b is None:
            if market_a in market_data:
                price_a = market_data[market_a].get("mark_price", market_data[market_a].get("price"))
            if market_b in market_data:
                price_b = market_data[market_b].get("mark_price", market_data[market_b].get("price"))
        
        if price_a is None or price_b is None:
            return None
        
        if price_a > 0:
            spread_pct = (price_a - price_b) / price_a
            spread_bps = spread_pct * 10000
        else:
            spread_bps = 0
        
        funding_diff_bps = (funding_a - funding_b) * 10000 * 24 * 365
        
        return {
            "market_a": market_a,
            "market_b": market_b,
            "price_a": price_a,
            "price_b": price_b,
            "spread_pct": spread_pct if price_a > 0 else 0,
            "spread_bps": spread_bps,
            "funding_a": funding_a,
            "funding_b": funding_b,
            "funding_diff_apy_bps": funding_diff_bps
        }
    
    def _get_position(self, long_market: str, short_market: str) -> Optional[SpreadPosition]:
        for pos in self.positions:
            if pos.long_market == long_market and pos.short_market == short_market:
                return pos
            if pos.long_market == short_market and pos.short_market == long_market:
                return pos
        return None
    
    def _calculate_position_pnl(self, position: SpreadPosition) -> float:
        spread_change_bps = position.current_spread - position.entry_spread
        pnl = position.size * (spread_change_bps / 10000)
        return pnl
    
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action.get("action_type")
        params = action.get("params", {})
        
        if action_type == "open_spread":
            return self._open_spread_position(params)
        elif action_type == "close_spread":
            return self._close_spread_position(params)
        
        return {"success": False, "error": f"Unknown action: {action_type}"}
    
    def _open_spread_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        long_market = params["long_market"]
        short_market = params["short_market"]
        spread_bps = params["spread_bps"]
        size_usd = params.get("size_usd", self.max_position_usd)
        
        if Config.is_simulation():
            position = SpreadPosition(
                long_market=long_market,
                short_market=short_market,
                size=size_usd,
                entry_spread=spread_bps,
                current_spread=spread_bps,
                unrealized_pnl=0.0,
                opened_at=time.time()
            )
            self.positions.append(position)
            
            logger.info(f"[perp_spread] Opened spread: long {long_market}, short {short_market}, spread={spread_bps:.1f}bps")
            
            return {
                "success": True,
                "simulated": True,
                "long_market": long_market,
                "short_market": short_market,
                "size": size_usd,
                "entry_spread": spread_bps
            }
        
        return {"success": False, "error": "Live trading not implemented for spread strategy"}
    
    def _close_spread_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        long_market = params["long_market"]
        short_market = params["short_market"]
        
        position = self._get_position(long_market, short_market)
        if not position:
            return {"success": False, "error": "Position not found"}
        
        realized_pnl = self._calculate_position_pnl(position)
        self.total_pnl += realized_pnl
        
        self.pnl_history.append({
            "timestamp": time.time(),
            "long_market": position.long_market,
            "short_market": position.short_market,
            "entry_spread": position.entry_spread,
            "exit_spread": position.current_spread,
            "pnl": realized_pnl,
            "hold_time": time.time() - position.opened_at
        })
        
        self.positions.remove(position)
        
        logger.info(f"[perp_spread] Closed spread: long {long_market}, short {short_market}, PnL=${realized_pnl:.2f}")
        
        return {
            "success": True,
            "simulated": Config.is_simulation(),
            "realized_pnl": realized_pnl,
            "total_pnl": self.total_pnl
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "perp_spread",
            "enabled": True,
            "positions_count": len(self.positions),
            "positions": [
                {
                    "long_market": p.long_market,
                    "short_market": p.short_market,
                    "size": p.size,
                    "entry_spread": p.entry_spread,
                    "current_spread": p.current_spread,
                    "unrealized_pnl": p.unrealized_pnl,
                    "hold_time": time.time() - p.opened_at
                }
                for p in self.positions
            ],
            "total_pnl": self.total_pnl,
            "recent_trades": self.pnl_history[-10:]
        }
