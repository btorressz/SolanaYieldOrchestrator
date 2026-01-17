import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class PriorityProfile(Enum):
    CHEAP = "cheap"
    BALANCED = "balanced"
    FAST = "fast"


class ExecutionMode(Enum):
    PASSIVE = "passive"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


EXECUTION_MODE_CONFIG = {
    ExecutionMode.PASSIVE: {
        "priority_profile": PriorityProfile.CHEAP,
        "slippage_tolerance_bps": 25,
        "twap_slices": 20,
        "twap_interval_sec": 30,
        "description": "Lower fees, smaller slippage, more TWAP slices"
    },
    ExecutionMode.BALANCED: {
        "priority_profile": PriorityProfile.BALANCED,
        "slippage_tolerance_bps": 50,
        "twap_slices": 10,
        "twap_interval_sec": 60,
        "description": "Standard execution settings"
    },
    ExecutionMode.AGGRESSIVE: {
        "priority_profile": PriorityProfile.FAST,
        "slippage_tolerance_bps": 100,
        "twap_slices": 5,
        "twap_interval_sec": 120,
        "description": "Higher fees, larger slippage tolerance, fewer TWAP slices"
    }
}


_current_execution_mode = ExecutionMode.BALANCED


def get_execution_mode() -> ExecutionMode:
    global _current_execution_mode
    return _current_execution_mode


def set_execution_mode(mode: ExecutionMode):
    global _current_execution_mode
    _current_execution_mode = mode


def get_execution_config() -> dict:
    return EXECUTION_MODE_CONFIG[get_execution_mode()]

class ActionType(Enum):
    SWAP = "swap"
    OPEN_PERP = "open_perp"
    CLOSE_PERP = "close_perp"
    DEPOSIT = "deposit"
    WITHDRAW = "withdraw"

@dataclass
class Action:
    action_type: ActionType
    params: Dict[str, Any]
    priority_profile: PriorityProfile = PriorityProfile.BALANCED
    max_slippage_bps: int = 50
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class ExecutionResult:
    success: bool
    action: Action
    signature: Optional[str] = None
    error: Optional[str] = None
    simulated: bool = False
    execution_time_ms: float = 0.0
    fees_paid: float = 0.0

class PriorityRouter:
    def __init__(self, solana_client=None, jupiter_client=None, drift_client=None):
        self.solana_client = solana_client
        self.jupiter_client = jupiter_client
        self.drift_client = drift_client
        self._pending_actions: List[Action] = []
        self._execution_history: List[ExecutionResult] = []
    
    def get_priority_fee(self, profile: PriorityProfile) -> int:
        return Config.get_priority_fee(profile.value)
    
    def queue_action(self, action: Action):
        self._pending_actions.append(action)
        logger.info(f"Queued action: {action.action_type.value}")
    
    def get_pending_actions(self) -> List[Action]:
        return self._pending_actions.copy()
    
    def clear_pending_actions(self):
        self._pending_actions.clear()
    
    async def execute_action(self, action: Action) -> ExecutionResult:
        start_time = time.time()
        priority_fee = self.get_priority_fee(action.priority_profile)
        
        if Config.is_simulation():
            result = await self._simulate_action(action)
        else:
            result = await self._execute_live_action(action, priority_fee)
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        self._execution_history.append(result)
        
        return result
    
    async def _simulate_action(self, action: Action) -> ExecutionResult:
        logger.info(f"[SIMULATION] Executing {action.action_type.value}: {action.params}")
        
        await self._async_sleep(0.1)
        
        return ExecutionResult(
            success=True,
            action=action,
            signature=f"SIM_{action.action_type.value}_{int(time.time() * 1000)}",
            simulated=True
        )
    
    async def _execute_live_action(self, action: Action, priority_fee: int) -> ExecutionResult:
        try:
            if action.action_type == ActionType.SWAP:
                return await self._execute_swap(action, priority_fee)
            elif action.action_type == ActionType.OPEN_PERP:
                return await self._execute_open_perp(action, priority_fee)
            elif action.action_type == ActionType.CLOSE_PERP:
                return await self._execute_close_perp(action, priority_fee)
            else:
                return ExecutionResult(
                    success=False,
                    action=action,
                    error=f"Unsupported action type: {action.action_type.value}"
                )
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return ExecutionResult(
                success=False,
                action=action,
                error=str(e)
            )
    
    async def _execute_swap(self, action: Action, priority_fee: int) -> ExecutionResult:
        if not self.jupiter_client or not self.solana_client:
            return ExecutionResult(
                success=False,
                action=action,
                error="Jupiter or Solana client not initialized"
            )
        
        params = action.params
        input_mint = params.get("input_mint")
        output_mint = params.get("output_mint")
        amount = params.get("amount")
        
        quote = self.jupiter_client.get_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=action.max_slippage_bps
        )
        
        if not quote:
            return ExecutionResult(
                success=False,
                action=action,
                error="Failed to get swap quote"
            )
        
        user_pubkey = self.solana_client.get_pubkey()
        if not user_pubkey:
            return ExecutionResult(
                success=False,
                action=action,
                error="No wallet connected"
            )
        
        swap_tx = self.jupiter_client.get_swap_transaction(quote, str(user_pubkey))
        if not swap_tx:
            return ExecutionResult(
                success=False,
                action=action,
                error="Failed to build swap transaction"
            )
        
        return ExecutionResult(
            success=True,
            action=action,
            signature="SWAP_TX_PLACEHOLDER"
        )
    
    async def _execute_open_perp(self, action: Action, priority_fee: int) -> ExecutionResult:
        if not self.drift_client:
            return ExecutionResult(
                success=False,
                action=action,
                error="Drift client not initialized"
            )
        
        from .drift_client import PositionDirection
        
        params = action.params
        market_index = params.get("market_index")
        size = params.get("size")
        direction_str = params.get("direction", "long")
        direction = PositionDirection.LONG if direction_str == "long" else PositionDirection.SHORT
        
        result = await self.drift_client.open_perp_position(
            market_index=market_index,
            size=size,
            direction=direction
        )
        
        return ExecutionResult(
            success=result.get("success", False),
            action=action,
            signature=result.get("signature"),
            error=result.get("error"),
            simulated=result.get("simulated", False)
        )
    
    async def _execute_close_perp(self, action: Action, priority_fee: int) -> ExecutionResult:
        if not self.drift_client:
            return ExecutionResult(
                success=False,
                action=action,
                error="Drift client not initialized"
            )
        
        params = action.params
        market_index = params.get("market_index")
        
        result = await self.drift_client.close_perp_position(market_index)
        
        return ExecutionResult(
            success=result.get("success", False),
            action=action,
            signature=result.get("signature"),
            error=result.get("error"),
            simulated=result.get("simulated", False)
        )
    
    async def execute_all_pending(self) -> List[ExecutionResult]:
        results = []
        for action in self._pending_actions:
            result = await self.execute_action(action)
            results.append(result)
        self._pending_actions.clear()
        return results
    
    def get_execution_history(self, limit: int = 100) -> List[ExecutionResult]:
        return self._execution_history[-limit:]
    
    async def _async_sleep(self, seconds: float):
        import asyncio
        await asyncio.sleep(seconds)


def create_swap_action(
    input_mint: str,
    output_mint: str,
    amount: int,
    priority_profile: PriorityProfile = PriorityProfile.BALANCED,
    max_slippage_bps: int = 50
) -> Action:
    return Action(
        action_type=ActionType.SWAP,
        params={
            "input_mint": input_mint,
            "output_mint": output_mint,
            "amount": amount
        },
        priority_profile=priority_profile,
        max_slippage_bps=max_slippage_bps
    )


def create_open_perp_action(
    market_index: int,
    size: float,
    direction: str,
    priority_profile: PriorityProfile = PriorityProfile.BALANCED
) -> Action:
    return Action(
        action_type=ActionType.OPEN_PERP,
        params={
            "market_index": market_index,
            "size": size,
            "direction": direction
        },
        priority_profile=priority_profile
    )


def create_close_perp_action(
    market_index: int,
    priority_profile: PriorityProfile = PriorityProfile.BALANCED
) -> Action:
    return Action(
        action_type=ActionType.CLOSE_PERP,
        params={
            "market_index": market_index
        },
        priority_profile=priority_profile
    )


@dataclass
class TWAPSlice:
    slice_number: int
    total_slices: int
    size: float
    executed_price: float
    timestamp: float
    success: bool
    signature: Optional[str] = None


@dataclass
class TWAPResult:
    success: bool
    total_slices: int
    executed_slices: int
    failed_slices: int
    total_size: float
    executed_size: float
    avg_price: float
    vwap: float
    slices: List[TWAPSlice]
    duration_sec: float
    simulated: bool = True
    
    @property
    def execution_pct(self) -> float:
        return (self.executed_size / self.total_size * 100) if self.total_size > 0 else 0


class TWAPExecutor:
    def __init__(self, priority_router: PriorityRouter, jupiter_client=None):
        self.priority_router = priority_router
        self.jupiter_client = jupiter_client
        self._active_twaps: Dict[str, Dict] = {}
    
    async def execute_twap_swap(
        self,
        input_mint: str,
        output_mint: str,
        total_amount: int,
        duration_sec: int,
        num_slices: int,
        priority_profile: str = "balanced",
        max_slippage_bps: int = 50,
        simulate: bool = True
    ) -> TWAPResult:
        slice_amount = total_amount // num_slices
        slice_interval = duration_sec / num_slices
        
        slices: List[TWAPSlice] = []
        executed_size = 0.0
        total_value = 0.0
        executed_slices = 0
        failed_slices = 0
        
        start_time = time.time()
        twap_id = f"twap_{int(start_time * 1000)}"
        
        self._active_twaps[twap_id] = {
            "status": "running",
            "total_slices": num_slices,
            "completed": 0
        }
        
        logger.info(f"[TWAP] Starting {num_slices} slices over {duration_sec}s for {total_amount} units")
        
        for i in range(num_slices):
            current_slice_amount = slice_amount
            if i == num_slices - 1:
                current_slice_amount = total_amount - (slice_amount * (num_slices - 1))
            
            try:
                if simulate:
                    base_price = 100.0
                    price_variance = (i - num_slices/2) * 0.001
                    simulated_price = base_price * (1 + price_variance)
                    
                    slice_result = TWAPSlice(
                        slice_number=i + 1,
                        total_slices=num_slices,
                        size=current_slice_amount / 1e9,
                        executed_price=simulated_price,
                        timestamp=time.time(),
                        success=True,
                        signature=f"SIM_TWAP_{twap_id}_{i+1}"
                    )
                    
                    executed_size += current_slice_amount / 1e9
                    total_value += (current_slice_amount / 1e9) * simulated_price
                    executed_slices += 1
                else:
                    action = create_swap_action(
                        input_mint=input_mint,
                        output_mint=output_mint,
                        amount=current_slice_amount,
                        priority_profile=PriorityProfile(priority_profile),
                        max_slippage_bps=max_slippage_bps
                    )
                    
                    result = await self.priority_router.execute_action(action)
                    
                    if result.success:
                        price = 100.0
                        slice_result = TWAPSlice(
                            slice_number=i + 1,
                            total_slices=num_slices,
                            size=current_slice_amount / 1e9,
                            executed_price=price,
                            timestamp=time.time(),
                            success=True,
                            signature=result.signature
                        )
                        executed_size += current_slice_amount / 1e9
                        total_value += (current_slice_amount / 1e9) * price
                        executed_slices += 1
                    else:
                        slice_result = TWAPSlice(
                            slice_number=i + 1,
                            total_slices=num_slices,
                            size=current_slice_amount / 1e9,
                            executed_price=0,
                            timestamp=time.time(),
                            success=False
                        )
                        failed_slices += 1
                
                slices.append(slice_result)
                
                self._active_twaps[twap_id]["completed"] = i + 1
                
                logger.debug(f"[TWAP] Slice {i+1}/{num_slices} executed @ {slice_result.executed_price:.4f}")
                
                if i < num_slices - 1:
                    await self._async_sleep(slice_interval)
                    
            except Exception as e:
                logger.error(f"[TWAP] Slice {i+1} failed: {e}")
                slices.append(TWAPSlice(
                    slice_number=i + 1,
                    total_slices=num_slices,
                    size=current_slice_amount / 1e9,
                    executed_price=0,
                    timestamp=time.time(),
                    success=False
                ))
                failed_slices += 1
        
        duration = time.time() - start_time
        vwap = total_value / executed_size if executed_size > 0 else 0
        avg_price = sum(s.executed_price for s in slices if s.success) / executed_slices if executed_slices > 0 else 0
        
        del self._active_twaps[twap_id]
        
        logger.info(f"[TWAP] Completed: {executed_slices}/{num_slices} slices, VWAP={vwap:.4f}")
        
        return TWAPResult(
            success=executed_slices > 0,
            total_slices=num_slices,
            executed_slices=executed_slices,
            failed_slices=failed_slices,
            total_size=total_amount / 1e9,
            executed_size=executed_size,
            avg_price=avg_price,
            vwap=vwap,
            slices=slices,
            duration_sec=duration,
            simulated=simulate
        )
    
    def simulate_twap_vs_single(
        self,
        total_amount: int,
        num_slices: int,
        price_series: List[float]
    ) -> Dict[str, Any]:
        if len(price_series) < num_slices:
            return {"error": "Insufficient price data for simulation"}
        
        single_shot_price = price_series[0]
        
        slice_indices = [int(i * (len(price_series) - 1) / (num_slices - 1)) for i in range(num_slices)]
        twap_prices = [price_series[idx] for idx in slice_indices]
        twap_vwap = sum(twap_prices) / len(twap_prices)
        
        slippage_single = 0.001 * (total_amount / 1e9)
        slippage_twap = 0.0003 * (total_amount / 1e9)
        
        effective_single = single_shot_price * (1 + slippage_single)
        effective_twap = twap_vwap * (1 + slippage_twap)
        
        improvement_bps = ((effective_single - effective_twap) / effective_single) * 10000
        
        return {
            "single_shot": {
                "price": single_shot_price,
                "estimated_slippage": slippage_single,
                "effective_price": effective_single
            },
            "twap": {
                "vwap": twap_vwap,
                "num_slices": num_slices,
                "estimated_slippage": slippage_twap,
                "effective_price": effective_twap
            },
            "improvement_bps": improvement_bps,
            "recommendation": "twap" if improvement_bps > 5 else "single_shot"
        }
    
    def get_active_twaps(self) -> Dict[str, Dict]:
        return self._active_twaps.copy()
    
    async def _async_sleep(self, seconds: float):
        import asyncio
        await asyncio.sleep(seconds)


def twap_result_to_dict(result: TWAPResult) -> Dict[str, Any]:
    return {
        "success": result.success,
        "total_slices": result.total_slices,
        "executed_slices": result.executed_slices,
        "failed_slices": result.failed_slices,
        "total_size": result.total_size,
        "executed_size": result.executed_size,
        "execution_pct": result.execution_pct,
        "avg_price": result.avg_price,
        "vwap": result.vwap,
        "duration_sec": result.duration_sec,
        "simulated": result.simulated,
        "slices": [
            {
                "slice_number": s.slice_number,
                "size": s.size,
                "executed_price": s.executed_price,
                "success": s.success,
                "timestamp": s.timestamp
            }
            for s in result.slices
        ]
    }
