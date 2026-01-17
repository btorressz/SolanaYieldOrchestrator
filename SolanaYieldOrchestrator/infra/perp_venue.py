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
        raise NotImplementedError

    @abstractmethod
    def get_mark_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def get_index_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def get_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_all_funding_rates(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_market_data(self, symbol: str) -> Optional[PerpMarketData]:
        raise NotImplementedError

    @abstractmethod
    def get_all_markets(self) -> List[PerpMarketData]:
        raise NotImplementedError

    @abstractmethod
    def open_position(
        self,
        symbol: str,
        size: float,
        side: str,
        reduce_only: bool = False,
        **kwargs: Any,
    ) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        size: Optional[float] = None,
        **kwargs: Any,
    ) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[PerpPosition]:
        raise NotImplementedError

    @abstractmethod
    def get_account_value(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_available_margin(self) -> float:
        raise NotImplementedError

    def get_funding_apy(self, symbol: str) -> Optional[float]:
        funding = self.get_funding(symbol)
        if funding:
            rate = float(funding.get("rate", 0.0))
            return rate * 24.0 * 365.0 * 100.0
        return None

    def compare_funding_with(self, other_venue: "PerpVenue", symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for symbol in symbols:
            my_funding = self.get_funding(symbol)
            other_funding = other_venue.get_funding(symbol)

            if my_funding and other_funding:
                my_rate = float(my_funding.get("rate", 0.0))
                other_rate = float(other_funding.get("rate", 0.0))
                diff = my_rate - other_rate

                results[symbol] = {
                    f"{self.name.value}_rate": my_rate,
                    f"{other_venue.name.value}_rate": other_rate,
                    "rate_diff": diff,
                    "diff_bps": diff * 10000.0,
                    "suggested_action": self._suggest_arb_action(my_rate, other_rate),
                }

        return results

    def _suggest_arb_action(self, my_rate: float, other_rate: float, min_diff_bps: float = 5) -> Optional[str]:
        diff_bps = abs(my_rate - other_rate) * 10000.0
        if diff_bps < float(min_diff_bps):
            return None

        if my_rate > other_rate:
            return f"Short on {self.name.value}, Long on other venue"
        return f"Long on {self.name.value}, Short on other venue"


class VenueRegistry:
    _venues: Dict[VenueName, PerpVenue] = {}

    @classmethod
    def register(cls, venue: PerpVenue) -> None:
        cls._venues[venue.name] = venue

    @classmethod
    def get(cls, name: VenueName) -> Optional[PerpVenue]:
        return cls._venues.get(name)

    @classmethod
    def get_all(cls) -> List[PerpVenue]:
        return list(cls._venues.values())

    @classmethod
    def get_cross_venue_funding(cls, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        venues = cls.get_all()
        if len(venues) < 2:
            return {}

        results: Dict[str, Dict[str, Any]] = {}

        for symbol in symbols:
            # FIX for pyright Ln 201 / 203:
            # Explicitly type 'venues' as Dict[str, Dict[str, float]] so assignment expects a dict, not Unknown.
            symbol_venues: Dict[str, Dict[str, float]] = {}
            symbol_data: Dict[str, Any] = {"venues": symbol_venues}

            for venue in venues:
                funding = venue.get_funding(symbol)
                if funding:
                    rate = float(funding.get("rate", 0.0))
                    symbol_venues[venue.name.value] = {
                        "rate": rate,
                        "apy": rate * 24.0 * 365.0 * 100.0,
                    }

            if len(symbol_venues) >= 2:
                rates = [v["rate"] for v in symbol_venues.values()]
                symbol_data["max_diff_bps"] = (max(rates) - min(rates)) * 10000.0

                venue_rates = list(symbol_venues.items())
                if len(venue_rates) >= 2:
                    v1_name, v1_data = venue_rates[0]
                    v该 = None  # keep lint quiet if a linter complains; not required by pyright
                    v2_name, v2_data = venue_rates[1]

                    if float(v1_data["rate"]) > float(v2_data["rate"]):
                        symbol_data["suggested"] = f"Short {v1_name}, Long {v2_name}"
                    else:
                        symbol_data["suggested"] = f"Short {v2_name}, Long {v1_name}"

            results[symbol] = symbol_data

        return results
