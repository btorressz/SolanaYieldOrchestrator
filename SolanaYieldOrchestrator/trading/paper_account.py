"""
PaperAccount - Simulated trading account for tracking virtual positions and PnL.
Supports both spot (Jupiter) and perpetual (Drift) positions.
Includes PnL component tracking: funding, basis, fees, and mark-to-market.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PositionType(Enum):
    SPOT = "spot"
    PERP = "perp"


class Side(Enum):
    LONG = "long"
    SHORT = "short"
    BUY = "buy"
    SELL = "sell"


@dataclass
class PnLComponents:
    """Tracks PnL broken down by source for the entire account."""
    funding_pnl: float = 0.0
    basis_pnl: float = 0.0
    fees_pnl: float = 0.0
    mark_to_market_pnl: float = 0.0
    
    @property
    def total(self) -> float:
        return self.funding_pnl + self.basis_pnl - self.fees_pnl + self.mark_to_market_pnl
    
    @property
    def yield_only(self) -> float:
        """PnL from yield sources only (funding + basis)."""
        return self.funding_pnl + self.basis_pnl
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "funding": self.funding_pnl,
            "basis": self.basis_pnl,
            "fees": self.fees_pnl,
            "mark_to_market": self.mark_to_market_pnl,
            "yield_only": self.yield_only,
            "total": self.total
        }
    
    def reset(self):
        self.funding_pnl = 0.0
        self.basis_pnl = 0.0
        self.fees_pnl = 0.0
        self.mark_to_market_pnl = 0.0


@dataclass
class SpotPosition:
    market: str
    base_asset: str
    quote_asset: str
    size: float
    avg_entry_price: float
    current_price: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def notional_value(self) -> float:
        return self.size * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.size * (self.current_price - self.avg_entry_price)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return ((self.current_price / self.avg_entry_price) - 1) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "spot",
            "market": self.market,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "size": self.size,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "notional_value": self.notional_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "timestamp": self.timestamp
        }


@dataclass
class PerpPosition:
    market: str
    size: float
    side: str
    avg_entry_price: float
    leverage: float = 1.0
    margin_used: float = 0.0
    current_price: float = 0.0
    funding_paid: float = 0.0
    basis_captured: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def notional_value(self) -> float:
        return abs(self.size) * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        price_diff = self.current_price - self.avg_entry_price
        if self.side == "short":
            price_diff = -price_diff
        return abs(self.size) * price_diff - self.funding_paid
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.margin_used == 0:
            return 0.0
        return (self.unrealized_pnl / self.margin_used) * 100
    
    @property
    def margin_ratio(self) -> float:
        if self.notional_value == 0:
            return float('inf')
        equity = self.margin_used + self.unrealized_pnl
        return equity / self.notional_value
    
    @property
    def liquidation_price(self) -> float:
        if self.size == 0 or self.margin_used == 0:
            return 0.0
        maintenance_margin_ratio = 0.0625
        if self.side == "long":
            return self.avg_entry_price * (1 - (self.margin_used / self.notional_value) + maintenance_margin_ratio)
        else:
            return self.avg_entry_price * (1 + (self.margin_used / self.notional_value) - maintenance_margin_ratio)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "perp",
            "market": self.market,
            "size": self.size,
            "side": self.side,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "leverage": self.leverage,
            "margin_used": self.margin_used,
            "notional_value": self.notional_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "margin_ratio": self.margin_ratio,
            "liquidation_price": self.liquidation_price,
            "funding_paid": self.funding_paid,
            "basis_captured": self.basis_captured,
            "timestamp": self.timestamp
        }


@dataclass
class TradeRecord:
    trade_id: str
    venue: str
    market: str
    side: str
    size: float
    price: float
    notional: float
    fees: float
    slippage_bps: float
    timestamp: float
    simulated: bool = True
    tx_signature: Optional[str] = None
    pnl_type: str = "mark_to_market"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "venue": self.venue,
            "market": self.market,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "notional": self.notional,
            "fees": self.fees,
            "slippage_bps": self.slippage_bps,
            "timestamp": self.timestamp,
            "simulated": self.simulated,
            "tx_signature": self.tx_signature,
            "pnl_type": self.pnl_type
        }


class PaperAccount:
    def __init__(self, initial_nav: float = 10000.0, account_id: str = "default"):
        self.account_id = account_id
        self.initial_nav = initial_nav
        self.cash_balance = initial_nav
        self.spot_positions: Dict[str, SpotPosition] = {}
        self.perp_positions: Dict[str, PerpPosition] = {}
        self.trade_history: List[TradeRecord] = []
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0
        self.created_at = time.time()
        self._trade_counter = 0
        self._pnl_components = PnLComponents()
        self._pnl_component_history: List[Dict[str, Any]] = []
        self._exposure_history: List[Dict[str, float]] = []
        
        logger.info(f"[PaperAccount] Initialized with NAV=${initial_nav:,.2f}")
    
    def _generate_trade_id(self) -> str:
        self._trade_counter += 1
        return f"{self.account_id}-{int(time.time())}-{self._trade_counter}"
    
    @property
    def total_spot_value(self) -> float:
        return sum(pos.notional_value for pos in self.spot_positions.values())
    
    @property
    def total_perp_notional(self) -> float:
        return sum(pos.notional_value for pos in self.perp_positions.values())
    
    @property
    def total_perp_margin(self) -> float:
        return sum(pos.margin_used for pos in self.perp_positions.values())
    
    @property
    def total_unrealized_pnl(self) -> float:
        spot_pnl = sum(pos.unrealized_pnl for pos in self.spot_positions.values())
        perp_pnl = sum(pos.unrealized_pnl for pos in self.perp_positions.values())
        return spot_pnl + perp_pnl
    
    @property
    def current_nav(self) -> float:
        return self.cash_balance + self.total_spot_value + self.total_perp_margin + self.total_unrealized_pnl
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.total_unrealized_pnl
    
    @property
    def total_return_pct(self) -> float:
        if self.initial_nav == 0:
            return 0.0
        return ((self.current_nav / self.initial_nav) - 1) * 100
    
    @property
    def available_margin(self) -> float:
        return self.cash_balance - self.total_perp_margin
    
    def get_pnl_breakdown(self) -> Dict[str, float]:
        """Get PnL broken down by component (funding, basis, fees, mark_to_market)."""
        return self._pnl_components.to_dict()
    
    def record_pnl_component(self, component_type: str, amount: float, market: str = None):
        """Record a PnL contribution by type."""
        if component_type == "funding":
            self._pnl_components.funding_pnl += amount
        elif component_type == "basis":
            self._pnl_components.basis_pnl += amount
        elif component_type == "fees":
            self._pnl_components.fees_pnl += abs(amount)
        elif component_type == "mark_to_market":
            self._pnl_components.mark_to_market_pnl += amount
        
        self._pnl_component_history.append({
            "timestamp": time.time(),
            "component": component_type,
            "amount": amount,
            "market": market
        })
        
        if len(self._pnl_component_history) > 5000:
            self._pnl_component_history = self._pnl_component_history[-2500:]
    
    def get_pnl_component_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get PnL component history for charting."""
        cutoff = time.time() - (hours * 3600)
        return [h for h in self._pnl_component_history if h["timestamp"] >= cutoff]
    
    def record_exposure(self):
        """Record current exposure by market for max exposure tracking."""
        exposures = {}
        for market, pos in self.spot_positions.items():
            exposures[market] = pos.notional_value
        for market, pos in self.perp_positions.items():
            exposures[market] = pos.notional_value
        
        self._exposure_history.append(exposures)
        
        if len(self._exposure_history) > 1000:
            self._exposure_history = self._exposure_history[-500:]
    
    def get_max_exposure_by_market(self) -> Dict[str, float]:
        """Get maximum exposure seen per market."""
        max_exposures = {}
        for entry in self._exposure_history:
            for market, exposure in entry.items():
                current_max = max_exposures.get(market, 0)
                max_exposures[market] = max(current_max, abs(exposure))
        
        for market, pos in self.spot_positions.items():
            current_max = max_exposures.get(market, 0)
            max_exposures[market] = max(current_max, abs(pos.notional_value))
        for market, pos in self.perp_positions.items():
            current_max = max_exposures.get(market, 0)
            max_exposures[market] = max(current_max, abs(pos.notional_value))
        
        return max_exposures
    
    def update_prices(self, prices: Dict[str, float]):
        for market, position in self.spot_positions.items():
            base_asset = position.base_asset.upper()
            if base_asset in prices:
                position.current_price = prices[base_asset]
        
        for market, position in self.perp_positions.items():
            base_symbol = market.replace("-PERP", "").upper()
            if base_symbol in prices:
                position.current_price = prices[base_symbol]
    
    def execute_spot_trade(
        self,
        market: str,
        side: str,
        size: float,
        price: float,
        fees: float = 0.0,
        slippage_bps: float = 0.0,
        simulated: bool = True,
        tx_signature: Optional[str] = None,
        pnl_type: str = "mark_to_market"
    ) -> Dict[str, Any]:
        parts = market.split("/")
        base_asset = parts[0] if len(parts) > 0 else market
        quote_asset = parts[1] if len(parts) > 1 else "USDC"
        
        notional = size * price
        total_cost = notional + fees
        
        if side.lower() in ["buy", "long"]:
            if total_cost > self.cash_balance:
                return {
                    "success": False,
                    "error": f"Insufficient cash balance. Required: ${total_cost:.2f}, Available: ${self.cash_balance:.2f}"
                }
            
            self.cash_balance -= total_cost
            
            if market in self.spot_positions:
                existing = self.spot_positions[market]
                total_size = existing.size + size
                avg_price = ((existing.size * existing.avg_entry_price) + (size * price)) / total_size
                existing.size = total_size
                existing.avg_entry_price = avg_price
                existing.current_price = price
            else:
                self.spot_positions[market] = SpotPosition(
                    market=market,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    size=size,
                    avg_entry_price=price,
                    current_price=price
                )
        else:
            if market not in self.spot_positions or self.spot_positions[market].size < size:
                available = self.spot_positions[market].size if market in self.spot_positions else 0
                return {
                    "success": False,
                    "error": f"Insufficient position size. Required: {size}, Available: {available}"
                }
            
            position = self.spot_positions[market]
            pnl = size * (price - position.avg_entry_price) - fees
            self.realized_pnl += pnl
            self.cash_balance += notional - fees
            
            self.record_pnl_component(pnl_type, pnl, market)
            
            position.size -= size
            if position.size <= 0.0001:
                del self.spot_positions[market]
        
        self.total_fees_paid += fees
        self.record_pnl_component("fees", fees, market)
        self.record_exposure()
        
        trade = TradeRecord(
            trade_id=self._generate_trade_id(),
            venue="jupiter",
            market=market,
            side=side,
            size=size,
            price=price,
            notional=notional,
            fees=fees,
            slippage_bps=slippage_bps,
            timestamp=time.time(),
            simulated=simulated,
            tx_signature=tx_signature,
            pnl_type=pnl_type
        )
        self.trade_history.append(trade)
        
        logger.info(f"[PaperAccount] Spot {side} {size} {market} @ ${price:.4f}")
        
        return {
            "success": True,
            "trade": trade.to_dict(),
            "position": self.spot_positions.get(market, {}).to_dict() if market in self.spot_positions else None,
            "cash_balance": self.cash_balance,
            "nav": self.current_nav
        }
    
    def execute_perp_trade(
        self,
        market: str,
        side: str,
        size: float,
        price: float,
        leverage: float = 1.0,
        fees: float = 0.0,
        slippage_bps: float = 0.0,
        simulated: bool = True,
        tx_signature: Optional[str] = None,
        pnl_type: str = "mark_to_market"
    ) -> Dict[str, Any]:
        notional = size * price
        margin_required = notional / leverage
        
        is_opening = side.lower() in ["long", "short"]
        is_closing = side.lower() in ["close_long", "close_short", "close"]
        
        if is_opening:
            if margin_required + fees > self.available_margin:
                return {
                    "success": False,
                    "error": f"Insufficient margin. Required: ${margin_required + fees:.2f}, Available: ${self.available_margin:.2f}"
                }
            
            normalized_side = "long" if side.lower() == "long" else "short"
            
            if market in self.perp_positions:
                existing = self.perp_positions[market]
                if existing.side == normalized_side:
                    total_size = existing.size + size
                    avg_price = ((existing.size * existing.avg_entry_price) + (size * price)) / total_size
                    existing.size = total_size
                    existing.avg_entry_price = avg_price
                    existing.margin_used += margin_required
                    existing.current_price = price
                else:
                    if size >= existing.size:
                        pnl = existing.unrealized_pnl - fees
                        self.realized_pnl += pnl
                        self.cash_balance += existing.margin_used
                        self.record_pnl_component(pnl_type, pnl, market)
                        remaining_size = size - existing.size
                        
                        if remaining_size > 0:
                            new_margin = (remaining_size * price) / leverage
                            self.cash_balance -= new_margin + fees
                            self.perp_positions[market] = PerpPosition(
                                market=market,
                                size=remaining_size,
                                side=normalized_side,
                                avg_entry_price=price,
                                leverage=leverage,
                                margin_used=new_margin,
                                current_price=price
                            )
                        else:
                            del self.perp_positions[market]
                    else:
                        pnl = (size * (price - existing.avg_entry_price)) * (-1 if existing.side == "short" else 1) - fees
                        self.realized_pnl += pnl
                        self.record_pnl_component(pnl_type, pnl, market)
                        margin_released = existing.margin_used * (size / existing.size)
                        existing.size -= size
                        existing.margin_used -= margin_released
                        self.cash_balance += margin_released
            else:
                self.cash_balance -= margin_required + fees
                self.perp_positions[market] = PerpPosition(
                    market=market,
                    size=size,
                    side=normalized_side,
                    avg_entry_price=price,
                    leverage=leverage,
                    margin_used=margin_required,
                    current_price=price
                )
        else:
            if market not in self.perp_positions:
                return {
                    "success": False,
                    "error": f"No position to close for {market}"
                }
            
            position = self.perp_positions[market]
            close_size = min(size, position.size)
            pnl = position.unrealized_pnl * (close_size / position.size) - fees
            margin_released = position.margin_used * (close_size / position.size)
            
            self.realized_pnl += pnl
            self.cash_balance += margin_released
            self.record_pnl_component(pnl_type, pnl, market)
            
            if close_size >= position.size:
                del self.perp_positions[market]
            else:
                position.size -= close_size
                position.margin_used -= margin_released
        
        self.total_fees_paid += fees
        self.record_pnl_component("fees", fees, market)
        self.record_exposure()
        
        trade = TradeRecord(
            trade_id=self._generate_trade_id(),
            venue="drift",
            market=market,
            side=side,
            size=size,
            price=price,
            notional=notional,
            fees=fees,
            slippage_bps=slippage_bps,
            timestamp=time.time(),
            simulated=simulated,
            tx_signature=tx_signature,
            pnl_type=pnl_type
        )
        self.trade_history.append(trade)
        
        logger.info(f"[PaperAccount] Perp {side} {size} {market} @ ${price:.4f} ({leverage}x)")
        
        return {
            "success": True,
            "trade": trade.to_dict(),
            "position": self.perp_positions.get(market, {}).to_dict() if market in self.perp_positions else None,
            "margin_used": self.total_perp_margin,
            "available_margin": self.available_margin,
            "nav": self.current_nav
        }
    
    def apply_funding_payment(self, market: str, funding_amount: float):
        """Apply funding payment and track it as funding PnL."""
        if market in self.perp_positions:
            position = self.perp_positions[market]
            position.funding_paid += funding_amount
            if funding_amount > 0:
                self.cash_balance -= funding_amount
            else:
                self.cash_balance += abs(funding_amount)
            
            funding_pnl = -funding_amount
            self.record_pnl_component("funding", funding_pnl, market)
            logger.debug(f"[PaperAccount] Applied funding ${funding_amount:.4f} to {market}")
    
    def record_basis_capture(self, market: str, basis_pnl: float):
        """Record PnL captured from basis convergence."""
        if market in self.perp_positions:
            self.perp_positions[market].basis_captured += basis_pnl
        
        self.record_pnl_component("basis", basis_pnl, market)
        logger.debug(f"[PaperAccount] Recorded basis PnL ${basis_pnl:.4f} for {market}")
    
    def check_liquidations(self, maintenance_margin_ratio: float = 0.0625) -> List[Dict[str, Any]]:
        liquidations = []
        for market, position in list(self.perp_positions.items()):
            if position.margin_ratio < maintenance_margin_ratio:
                liquidations.append({
                    "market": market,
                    "position": position.to_dict(),
                    "margin_ratio": position.margin_ratio,
                    "action": "liquidated"
                })
                self.realized_pnl -= position.margin_used
                del self.perp_positions[market]
                logger.warning(f"[PaperAccount] Position liquidated: {market}")
        return liquidations
    
    def partial_liquidation_check(self, warning_ratio: float = 0.10, trim_pct: float = 0.25) -> List[Dict[str, Any]]:
        warnings = []
        for market, position in self.perp_positions.items():
            if position.margin_ratio < warning_ratio:
                trim_size = position.size * trim_pct
                warnings.append({
                    "market": market,
                    "current_margin_ratio": position.margin_ratio,
                    "recommended_trim_size": trim_size,
                    "recommended_trim_pct": trim_pct * 100,
                    "severity": "high" if position.margin_ratio < 0.075 else "medium"
                })
        return warnings
    
    def reset(self, new_initial_nav: Optional[float] = None):
        nav = new_initial_nav if new_initial_nav is not None else self.initial_nav
        self.initial_nav = nav
        self.cash_balance = nav
        self.spot_positions.clear()
        self.perp_positions.clear()
        self.trade_history.clear()
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0
        self._trade_counter = 0
        self._pnl_components.reset()
        self._pnl_component_history.clear()
        self._exposure_history.clear()
        logger.info(f"[PaperAccount] Reset with NAV=${nav:,.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "initial_nav": self.initial_nav,
            "current_nav": self.current_nav,
            "cash_balance": self.cash_balance,
            "total_spot_value": self.total_spot_value,
            "total_perp_notional": self.total_perp_notional,
            "total_perp_margin": self.total_perp_margin,
            "available_margin": self.available_margin,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.total_unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "total_fees_paid": self.total_fees_paid,
            "pnl_breakdown": self.get_pnl_breakdown(),
            "spot_positions": [pos.to_dict() for pos in self.spot_positions.values()],
            "perp_positions": [pos.to_dict() for pos in self.perp_positions.values()],
            "trade_count": len(self.trade_history),
            "created_at": self.created_at
        }
    
    def get_positions_summary(self) -> Dict[str, Any]:
        return {
            "spot": [pos.to_dict() for pos in self.spot_positions.values()],
            "perp": [pos.to_dict() for pos in self.perp_positions.values()],
            "total_spot_value": self.total_spot_value,
            "total_perp_notional": self.total_perp_notional,
            "total_perp_margin": self.total_perp_margin,
            "unrealized_pnl": self.total_unrealized_pnl
        }
    
    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [trade.to_dict() for trade in self.trade_history[-limit:]]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        perp_risks = []
        for market, position in self.perp_positions.items():
            perp_risks.append({
                "market": market,
                "side": position.side,
                "size": position.size,
                "leverage": position.leverage,
                "margin_used": position.margin_used,
                "margin_ratio": position.margin_ratio,
                "liquidation_price": position.liquidation_price,
                "unrealized_pnl": position.unrealized_pnl,
                "risk_level": "high" if position.margin_ratio < 0.10 else "medium" if position.margin_ratio < 0.20 else "low"
            })
        
        partial_liq_warnings = self.partial_liquidation_check()
        
        return {
            "total_leverage": self.total_perp_notional / self.current_nav if self.current_nav > 0 else 0,
            "margin_usage_pct": (self.total_perp_margin / self.cash_balance * 100) if self.cash_balance > 0 else 0,
            "perp_positions": perp_risks,
            "liquidation_warnings": partial_liq_warnings,
            "overall_risk": "high" if any(p["risk_level"] == "high" for p in perp_risks) else "medium" if any(p["risk_level"] == "medium" for p in perp_risks) else "low",
            "pnl_breakdown": self.get_pnl_breakdown(),
            "max_exposure_by_market": self.get_max_exposure_by_market()
        }


paper_account_instance: Optional[PaperAccount] = None


def get_paper_account(initial_nav: float = 10000.0) -> PaperAccount:
    global paper_account_instance
    if paper_account_instance is None:
        paper_account_instance = PaperAccount(initial_nav=initial_nav)
    return paper_account_instance


def reset_paper_account(initial_nav: float = 10000.0) -> PaperAccount:
    global paper_account_instance
    paper_account_instance = PaperAccount(initial_nav=initial_nav)
    return paper_account_instance
