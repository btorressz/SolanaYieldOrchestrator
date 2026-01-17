import asyncio
import threading
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ChainState:
    latest_slot: int = 0
    last_update_ts: float = 0.0
    recent_logs: List[Dict[str, Any]] = field(default_factory=list)
    is_connected: bool = False
    connection_errors: int = 0


@dataclass
class TransactionInfo:
    signature: str
    slot: int
    block_time: Optional[int]
    success: bool
    fee: int
    compute_units: int
    signers: List[str]
    program_ids: List[str]
    instructions_count: int
    error: Optional[str] = None


class SolanaChainMonitor:
    def __init__(self, max_logs: int = 50):
        self.max_logs = max_logs
        self._state = ChainState()
        self._lock = threading.Lock()
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._callbacks: List[Callable] = []
        
        self._recent_txs: deque = deque(maxlen=100)
        
        self._priority_fees_cache: Dict[str, Any] = {}
        self._priority_fees_ts: float = 0
    
    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "latest_slot": self._state.latest_slot,
                "last_update_ts": self._state.last_update_ts,
                "seconds_since_update": time.time() - self._state.last_update_ts if self._state.last_update_ts else None,
                "is_connected": self._state.is_connected,
                "recent_logs_count": len(self._state.recent_logs),
                "recent_logs": self._state.recent_logs[-10:],
                "connection_errors": self._state.connection_errors
            }
    
    def _update_slot(self, slot: int):
        with self._lock:
            self._state.latest_slot = slot
            self._state.last_update_ts = time.time()
    
    def _add_log(self, log_entry: Dict[str, Any]):
        with self._lock:
            self._state.recent_logs.append(log_entry)
            if len(self._state.recent_logs) > self.max_logs:
                self._state.recent_logs = self._state.recent_logs[-self.max_logs:]
    
    async def _connect_websocket(self):
        try:
            from solana.rpc.websocket_api import connect
            
            async with connect(Config.SOLANA_WS_URL) as ws:
                with self._lock:
                    self._state.is_connected = True
                    self._state.connection_errors = 0
                
                logger.info("Solana WebSocket connected")
                
                await ws.slot_subscribe()
                first_resp = await ws.recv()
                slot_sub_id = first_resp[0].result
                
                async for msg in ws:
                    if not self._running:
                        break
                    
                    try:
                        if hasattr(msg, '__iter__'):
                            for item in msg:
                                if hasattr(item, 'result') and isinstance(item.result, dict):
                                    if 'slot' in item.result:
                                        self._update_slot(item.result['slot'])
                    except Exception as e:
                        logger.debug(f"Error processing WS message: {e}")
                
                await ws.slot_unsubscribe(slot_sub_id)
                
        except ImportError:
            logger.warning("solana-py websocket not available, using polling fallback")
            await self._polling_fallback()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            with self._lock:
                self._state.is_connected = False
                self._state.connection_errors += 1
            
            if self._running:
                await asyncio.sleep(5)
                await self._connect_websocket()
    
    async def _polling_fallback(self):
        try:
            from solana.rpc.api import Client
            
            client = Client(Config.SOLANA_RPC_URL)
            
            with self._lock:
                self._state.is_connected = True
            
            while self._running:
                try:
                    slot_resp = client.get_slot()
                    if hasattr(slot_resp, 'value'):
                        self._update_slot(slot_resp.value)
                except Exception as e:
                    logger.debug(f"Polling error: {e}")
                
                await asyncio.sleep(1)
                
        except ImportError:
            logger.warning("solana-py not available for chain monitoring")
            self._mock_monitoring()
        except Exception as e:
            logger.error(f"Polling fallback error: {e}")
    
    def _mock_monitoring(self):
        import hashlib
        
        with self._lock:
            self._state.is_connected = True
            
            base_slot = 250000000
            time_factor = int(time.time()) // 2
            
            hash_input = f"slot_{time_factor}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
            slot_offset = hash_val % 1000
            
            self._state.latest_slot = base_slot + slot_offset
            self._state.last_update_ts = time.time()
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        
        def run_loop():
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                try:
                    self._loop.run_until_complete(self._connect_websocket())
                except Exception as e:
                    logger.error(f"Chain monitor loop error: {e}")
                finally:
                    try:
                        self._loop.close()
                    except Exception:
                        pass
                    self._loop = None
            except Exception as e:
                logger.error(f"Failed to create event loop: {e}")
                self._mock_monitoring()
        
        if Config.is_simulation():
            self._mock_monitoring()
            
            def mock_updater():
                while self._running:
                    try:
                        self._mock_monitoring()
                    except Exception as e:
                        logger.debug(f"Mock monitoring error: {e}")
                    time.sleep(2)
            
            thread = threading.Thread(target=mock_updater, daemon=True)
            thread.start()
        else:
            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
    
    def stop(self):
        self._running = False
        with self._lock:
            self._state.is_connected = False
    
    def get_transaction(self, signature: str) -> Optional[TransactionInfo]:
        try:
            from solana.rpc.api import Client
            from solders.signature import Signature
            
            client = Client(Config.SOLANA_RPC_URL)
            
            sig = Signature.from_string(signature)
            resp = client.get_transaction(
                sig,
                encoding="json",
                max_supported_transaction_version=0
            )
            
            if not resp.value:
                return None
            
            tx = resp.value
            meta = tx.transaction.meta if hasattr(tx, 'transaction') else None
            
            signers = []
            program_ids = []
            instructions_count = 0
            
            if hasattr(tx, 'transaction') and hasattr(tx.transaction, 'transaction'):
                message = tx.transaction.transaction.message
                if hasattr(message, 'account_keys'):
                    signers = [str(k) for k in message.account_keys[:1]]
                if hasattr(message, 'instructions'):
                    instructions_count = len(message.instructions)
                    for ix in message.instructions:
                        if hasattr(ix, 'program_id_index'):
                            program_ids.append(str(message.account_keys[ix.program_id_index]))
            
            return TransactionInfo(
                signature=signature,
                slot=tx.slot if hasattr(tx, 'slot') else 0,
                block_time=tx.block_time if hasattr(tx, 'block_time') else None,
                success=meta.err is None if meta else True,
                fee=meta.fee if meta else 0,
                compute_units=meta.compute_units_consumed if meta and hasattr(meta, 'compute_units_consumed') else 0,
                signers=signers,
                program_ids=list(set(program_ids)),
                instructions_count=instructions_count,
                error=str(meta.err) if meta and meta.err else None
            )
            
        except ImportError:
            logger.warning("solana-py not available for transaction lookup")
            return self._mock_transaction(signature)
        except Exception as e:
            logger.error(f"Failed to get transaction {signature}: {e}")
            return None
    
    def _mock_transaction(self, signature: str) -> TransactionInfo:
        import hashlib
        
        hash_val = int(hashlib.md5(signature.encode()).hexdigest()[:8], 16)
        
        return TransactionInfo(
            signature=signature,
            slot=250000000 + (hash_val % 10000),
            block_time=int(time.time()) - (hash_val % 3600),
            success=hash_val % 10 != 0,
            fee=5000 + (hash_val % 10000),
            compute_units=50000 + (hash_val % 150000),
            signers=["MockSigner1..."],
            program_ids=["JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"],
            instructions_count=1 + (hash_val % 5),
            error=None if hash_val % 10 != 0 else "Mock error"
        )
    
    def get_priority_fees(self, force_refresh: bool = False) -> Dict[str, Any]:
        if not force_refresh and self._priority_fees_cache:
            if time.time() - self._priority_fees_ts < 30:
                return self._priority_fees_cache
        
        try:
            from solana.rpc.api import Client
            
            client = Client(Config.SOLANA_RPC_URL)
            
            resp = client._provider.make_request(
                "getRecentPrioritizationFees",
                []
            )
            
            if resp and "result" in resp:
                fees = resp["result"]
                if fees:
                    priority_fees = [f["prioritizationFee"] for f in fees if f.get("prioritizationFee")]
                    
                    if priority_fees:
                        sorted_fees = sorted(priority_fees)
                        p25_idx = len(sorted_fees) // 4
                        p50_idx = len(sorted_fees) // 2
                        p75_idx = int(len(sorted_fees) * 0.75)
                        
                        result = {
                            "cheap": sorted_fees[p25_idx] if p25_idx < len(sorted_fees) else 1000,
                            "balanced": sorted_fees[p50_idx] if p50_idx < len(sorted_fees) else 10000,
                            "fast": sorted_fees[p75_idx] if p75_idx < len(sorted_fees) else 100000,
                            "samples": len(sorted_fees),
                            "timestamp": time.time()
                        }
                        
                        self._priority_fees_cache = result
                        self._priority_fees_ts = time.time()
                        return result
            
        except Exception as e:
            logger.debug(f"Failed to get priority fees: {e}")
        
        return {
            "cheap": Config.PRIORITY_FEE_CHEAP,
            "balanced": Config.PRIORITY_FEE_BALANCED,
            "fast": Config.PRIORITY_FEE_FAST,
            "samples": 0,
            "timestamp": time.time(),
            "source": "default"
        }
    
    def get_wallet_holdings(self, public_key: Optional[str] = None) -> Dict[str, Any]:
        if Config.is_simulation():
            return self._mock_wallet_holdings()
        
        try:
            from solana.rpc.api import Client
            from solders.pubkey import Pubkey
            
            client = Client(Config.SOLANA_RPC_URL)
            
            if not public_key:
                return {"error": "No public key configured", "holdings": []}
            
            owner = Pubkey.from_string(public_key)
            
            sol_resp = client.get_balance(owner)
            sol_balance = sol_resp.value / 1e9 if hasattr(sol_resp, 'value') else 0
            
            holdings = [{
                "mint": "SOL",
                "symbol": "SOL",
                "balance": sol_balance,
                "decimals": 9,
                "is_native": True
            }]
            
            try:
                token_resp = client.get_token_accounts_by_owner_json_parsed(
                    owner,
                    opts={"programId": Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")}
                )
                
                if hasattr(token_resp, 'value'):
                    for account in token_resp.value:
                        info = account.account.data.parsed["info"]
                        mint = info["mint"]
                        balance = float(info["tokenAmount"]["uiAmount"] or 0)
                        decimals = info["tokenAmount"]["decimals"]
                        
                        if balance > 0:
                            holdings.append({
                                "mint": mint,
                                "symbol": self._get_token_symbol(mint),
                                "balance": balance,
                                "decimals": decimals,
                                "is_native": False
                            })
                            
            except Exception as e:
                logger.debug(f"Failed to get SPL tokens: {e}")
            
            return {
                "owner": public_key,
                "holdings": holdings,
                "timestamp": time.time()
            }
            
        except ImportError:
            return self._mock_wallet_holdings()
        except Exception as e:
            logger.error(f"Failed to get wallet holdings: {e}")
            return {"error": str(e), "holdings": []}
    
    def _mock_wallet_holdings(self) -> Dict[str, Any]:
        return {
            "owner": "MockWallet...",
            "holdings": [
                {"mint": "SOL", "symbol": "SOL", "balance": 10.5, "decimals": 9, "is_native": True},
                {"mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "symbol": "USDC", "balance": 1000.0, "decimals": 6, "is_native": False},
            ],
            "timestamp": time.time(),
            "source": "mock"
        }
    
    def _get_token_symbol(self, mint: str) -> str:
        known = {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
            "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So": "mSOL",
            "So11111111111111111111111111111111111111112": "wSOL",
        }
        return known.get(mint, mint[:8] + "...")
    
    def add_recent_tx(self, signature: str, success: bool = True):
        with self._lock:
            self._recent_txs.append({
                "signature": signature,
                "success": success,
                "timestamp": time.time()
            })
    
    def get_recent_txs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent_txs)[-limit:]


chain_monitor = SolanaChainMonitor()
