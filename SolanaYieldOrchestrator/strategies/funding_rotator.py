import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import Config
from strategies import Strategy, Action
from data.analytics import Analytics, FundingAnalysis
from utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class FundingPosition:
    market: str
    market_index: int
    size: float
    direction: str
    entry_funding_rate: float
    entry_funding_apy: float
    entry_time: float
    accumulated_funding: float = 0.0

class FundingRotator(Strategy):
    def __init__(
        self,
        target_alloc: float = None,
        top_n_markets: int = None,
        min_apy_threshold: float = None,
        rotation_epoch_seconds: int = None
    ):
        self._target_allocation = target_alloc or Config.ALLOCATION_FUNDING_ROTATOR
        self.top_n_markets = top_n_markets or Config.FUNDING_TOP_N_MARKETS
        self.min_apy_threshold = min_apy_threshold or Config.FUNDING_MIN_APY_THRESHOLD
        self.rotation_epoch_seconds = rotation_epoch_seconds or Config.ROTATION_EPOCH_SECONDS
        
        self.analytics = Analytics()
        self.positions: Dict[str, FundingPosition] = {}
        
        self._current_snapshot = None
        self._last_update = 0
        self._last_rotation = 0
        self._total_funding_earned = 0.0
        self._rotation_count = 0
    
    def name(self) -> str:
        return "funding_rotator"
    
    def target_allocation(self) -> float:
        return self._target_allocation
    
    def update_state(self, market_snapshot) -> None:
        self._current_snapshot = market_snapshot
        self._last_update = time.time()
        
        self._update_accumulated_funding()
    
    def _update_accumulated_funding(self):
        if not self._current_snapshot:
            return
        
        for market, position in self.positions.items():
            funding_data = self._current_snapshot.funding_rates.get(market)
            if funding_data:
                funding_rate = funding_data.funding_rate if hasattr(funding_data, 'funding_rate') else funding_data.get('rate', 0)
                
                hours_held = (time.time() - position.entry_time) / 3600
                
                if position.direction == "short":
                    funding_earned = funding_rate * position.size * hours_held
                else:
                    funding_earned = -funding_rate * position.size * hours_held
                
                position.accumulated_funding = funding_earned
    
    def _get_funding_rankings(self) -> List[FundingAnalysis]:
        if not self._current_snapshot:
            return []
        
        funding_data = self._current_snapshot.funding_rates
        analyses = self.analytics.analyze_funding(funding_data)
        
        return self.analytics.rank_markets_by_funding(analyses, self.min_apy_threshold)
    
    def _should_rotate(self) -> bool:
        if self._last_rotation == 0:
            return True
        
        time_since_rotation = time.time() - self._last_rotation
        return time_since_rotation >= self.rotation_epoch_seconds
    
    def desired_actions(self, vault_state: Dict[str, Any]) -> List[Action]:
        actions = []
        
        if not self._current_snapshot:
            return actions
        
        if not self._should_rotate():
            return actions
        
        available_capital = vault_state.get("available_capital", 0) * self._target_allocation
        
        rankings = self._get_funding_rankings()
        top_markets = rankings[:self.top_n_markets]
        
        if not top_markets:
            return actions
        
        top_market_names = {m.market for m in top_markets}
        
        for market, position in list(self.positions.items()):
            if market not in top_market_names:
                logger.info(f"[{self.name()}] Closing position in {market}: no longer in top {self.top_n_markets}")
                
                actions.append(Action(
                    action_type="close_perp",
                    params={
                        "market_index": position.market_index,
                        "market": market
                    },
                    priority="balanced"
                ))
        
        position_size_per_market = available_capital / self.top_n_markets if self.top_n_markets > 0 else 0
        
        for analysis in top_markets:
            market = analysis.market
            
            if market in self.positions:
                continue
            
            if position_size_per_market < 50:
                continue
            
            market_index = self._get_market_index(market)
            
            direction = "short" if analysis.current_rate > 0 else "long"
            
            perp_price = self._current_snapshot.perp_prices.get(market, 100.0)
            size = position_size_per_market / perp_price if perp_price > 0 else 0
            
            if size > 0:
                logger.info(f"[{self.name()}] Opening position in {market}: {direction}, APY={analysis.estimated_apy:.2f}%")
                
                actions.append(Action(
                    action_type="open_perp",
                    params={
                        "market_index": market_index,
                        "market": market,
                        "size": size,
                        "direction": direction
                    },
                    priority="balanced"
                ))
        
        if actions:
            self._last_rotation = time.time()
            self._rotation_count += 1
        
        return actions
    
    def _get_market_index(self, market: str) -> int:
        market_indices = {
            "SOL-PERP": 0,
            "BTC-PERP": 1,
            "ETH-PERP": 2,
            "APT-PERP": 3,
            "ARB-PERP": 4,
        }
        return market_indices.get(market, 0)
    
    def apply_simulated_trade(self, action: Action, success: bool = True):
        if not success:
            return
        
        market = action.params.get("market", "SOL-PERP")
        
        if action.action_type == "open_perp":
            size = action.params.get("size", 0)
            direction = action.params.get("direction", "short")
            market_index = action.params.get("market_index", 0)
            
            funding_data = self._current_snapshot.funding_rates.get(market) if self._current_snapshot else None
            funding_rate = 0.0
            funding_apy = 0.0
            
            if funding_data:
                funding_rate = funding_data.funding_rate if hasattr(funding_data, 'funding_rate') else 0
                funding_apy = funding_data.funding_apy if hasattr(funding_data, 'funding_apy') else 0
            
            self.positions[market] = FundingPosition(
                market=market,
                market_index=market_index,
                size=size,
                direction=direction,
                entry_funding_rate=funding_rate,
                entry_funding_apy=funding_apy,
                entry_time=time.time()
            )
            
            logger.info(f"[{self.name()}] Opened simulated {direction} position in {market}: size={size}")
        
        elif action.action_type == "close_perp":
            if market in self.positions:
                position = self.positions[market]
                self._total_funding_earned += position.accumulated_funding
                del self.positions[market]
                logger.info(f"[{self.name()}] Closed simulated position in {market}: funding earned={position.accumulated_funding:.4f}")
    
    def get_status(self) -> Dict[str, Any]:
        rankings = self._get_funding_rankings()
        
        position_value = 0.0
        if self._current_snapshot:
            for market, position in self.positions.items():
                perp_price = self._current_snapshot.perp_prices.get(market, 0)
                position_value += position.size * perp_price
        
        accumulated_funding = sum(p.accumulated_funding for p in self.positions.values())
        
        top_markets = []
        for r in rankings[:self.top_n_markets]:
            top_markets.append({
                "market": r.market,
                "funding_apy": r.estimated_apy,
                "direction": r.direction
            })
        
        current_positions = []
        for market, pos in self.positions.items():
            current_positions.append({
                "market": market,
                "size": pos.size,
                "direction": pos.direction,
                "accumulated_funding": pos.accumulated_funding
            })
        
        return {
            "name": self.name(),
            "target_allocation": self._target_allocation,
            "top_n_markets": self.top_n_markets,
            "min_apy_threshold": self.min_apy_threshold,
            "rotation_epoch_seconds": self.rotation_epoch_seconds,
            "top_funding_markets": top_markets,
            "active_positions": len(self.positions),
            "current_positions": current_positions,
            "position_value_usd": position_value,
            "accumulated_funding": accumulated_funding,
            "total_funding_earned": self._total_funding_earned,
            "rotation_count": self._rotation_count,
            "time_until_next_rotation": max(0, self.rotation_epoch_seconds - (time.time() - self._last_rotation)),
            "last_update": self._last_update
        }
