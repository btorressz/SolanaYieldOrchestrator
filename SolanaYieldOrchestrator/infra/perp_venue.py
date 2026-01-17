from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class VenueName(Enum):
    DRIFT = "drift"
    HYPERLIQUID = "hyperliquid"


@dataclass
class PerpPosition:
    symbol: str
    size: float
    side: str
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    margin_used: float
    leverage: float
    liquidation_price: Optional[float] = None


@dataclass
class PerpMarketData:
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    funding_rate_apy: float
    open_interest: float
    volume_24h: float
    next_funding_time: float


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    simulated: bool = False
    venue: Optional[str] = None


class PerpVenue(ABC):
    @property
    @abstractmethod
    def name(self) -> VenueName:
        pass
    
    @abstractmethod
    def get_mark_price(self, symbol: str) -> Optional[float]:
        pass
    
    @abstractmethod
    def get_index_price(self, symbol: str) -> Optional[float]:
        pass
    
    @abstractmethod
    def get_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_all_funding_rates(self) -> Dict[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str) -> Optional[PerpMarketData]:
        pass
    
    @abstractmethod
    def get_all_markets(self) -> List[PerpMarketData]:
        pass
    
    @abstractmethod
    def open_position(
        self,
        symbol: str,
        size: float,
        side: str,
        reduce_only: bool = False,
        **kwargs
    ) -> OrderResult:
        pass
    
    @abstractmethod
    def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,
        **kwargs
    ) -> OrderResult:
        pass
    
    @abstractmethod
    def get_positions(self) -> List[PerpPosition]:
        pass
    
    @abstractmethod
    def get_account_value(self) -> float:
        pass
    
    @abstractmethod
    def get_available_margin(self) -> float:
        pass
    
    def get_funding_apy(self, symbol: str) -> Optional[float]:
        funding = self.get_funding(symbol)
        if funding:
            rate = funding.get("rate", 0)
            return rate * 24 * 365 * 100
        return None
    
    def compare_funding_with(
        self,
        other_venue: "PerpVenue",
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        results = {}
        for symbol in symbols:
            my_funding = self.get_funding(symbol)
            other_funding = other_venue.get_funding(symbol)
            
            if my_funding and other_funding:
                my_rate = my_funding.get("rate", 0)
                other_rate = other_funding.get("rate", 0)
                diff = my_rate - other_rate
                
                results[symbol] = {
                    f"{self.name.value}_rate": my_rate,
                    f"{other_venue.name.value}_rate": other_rate,
                    "rate_diff": diff,
                    "diff_bps": diff * 10000,
                    "suggested_action": self._suggest_arb_action(my_rate, other_rate)
                }
        
        return results
    
    def _suggest_arb_action(
        self,
        my_rate: float,
        other_rate: float,
        min_diff_bps: float = 5
    ) -> Optional[str]:
        diff_bps = abs(my_rate - other_rate) * 10000
        if diff_bps < min_diff_bps:
            return None
        
        if my_rate > other_rate:
            return f"Short on {self.name.value}, Long on other venue"
        else:
            return f"Long on {self.name.value}, Short on other venue"


class VenueRegistry:
    _venues: Dict[VenueName, PerpVenue] = {}
    
    @classmethod
    def register(cls, venue: PerpVenue):
        cls._venues[venue.name] = venue
    
    @classmethod
    def get(cls, name: VenueName) -> Optional[PerpVenue]:
        return cls._venues.get(name)
    
    @classmethod
    def get_all(cls) -> List[PerpVenue]:
        return list(cls._venues.values())
    
    @classmethod
    def get_cross_venue_funding(
        cls,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        venues = cls.get_all()
        if len(venues) < 2:
            return {}
        
        results = {}
        for symbol in symbols:
            symbol_data = {"venues": {}}
            
            for venue in venues:
                funding = venue.get_funding(symbol)
                if funding:
                    symbol_data["venues"][venue.name.value] = {
                        "rate": funding.get("rate", 0),
                        "apy": funding.get("rate", 0) * 24 * 365 * 100
                    }
            
            if len(symbol_data["venues"]) >= 2:
                rates = [v["rate"] for v in symbol_data["venues"].values()]
                symbol_data["max_diff_bps"] = (max(rates) - min(rates)) * 10000
                
                venue_rates = list(symbol_data["venues"].items())
                if len(venue_rates) >= 2:
                    v1_name, v1_data = venue_rates[0]
                    v2_name, v2_data = venue_rates[1]
                    
                    if v1_data["rate"] > v2_data["rate"]:
                        symbol_data["suggested"] = f"Short {v1_name}, Long {v2_name}"
                    else:
                        symbol_data["suggested"] = f"Short {v2_name}, Long {v1_name}"
            
            results[symbol] = symbol_data
        
        return results
