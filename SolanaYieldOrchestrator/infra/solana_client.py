import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts, TokenAccountOpts

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solders.signature import Signature

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def decode_base58(data: str) -> bytes:
    ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    num = 0
    for char in data.encode():
        num = num * 58 + ALPHABET.index(char)

    result: List[int] = []
    while num > 0:
        result.append(num % 256)
        num //= 256

    for char in data:
        if char == "1":
            result.append(0)
        else:
            break

    return bytes(reversed(result))


@dataclass
class TransactionResult:
    success: bool
    signature: Optional[str] = None
    error: Optional[str] = None
    simulated: bool = False


class SolanaClient:
    def __init__(self):
        self.client = Client(Config.SOLANA_RPC_URL)
        self.keypair: Optional[Keypair] = None
        self._load_keypair()

    def _load_keypair(self) -> None:
        if getattr(Config, "SOLANA_PRIVATE_KEY", None):
            try:
                key_bytes = decode_base58(Config.SOLANA_PRIVATE_KEY)
                self.keypair = Keypair.from_bytes(key_bytes)
                logger.info(f"Loaded keypair from private key: {self.keypair.pubkey()}")
            except Exception as e:
                logger.warning(f"Failed to load keypair from private key: {e}")

        if not self.keypair and getattr(Config, "SOLANA_KEYPAIR_PATH", None):
            try:
                with open(Config.SOLANA_KEYPAIR_PATH, "r") as f:
                    key_data = json.load(f)
                self.keypair = Keypair.from_bytes(bytes(key_data))
                logger.info(f"Loaded keypair from file: {self.keypair.pubkey()}")
            except Exception as e:
                logger.warning(f"Failed to load keypair from file: {e}")

        if not self.keypair:
            logger.warning("No keypair loaded - running in read-only mode")

    def get_pubkey(self) -> Optional[Pubkey]:
        return self.keypair.pubkey() if self.keypair else None

    def get_balance(self, pubkey: Optional[Pubkey] = None) -> float:
        try:
            target = pubkey or self.get_pubkey()
            if not target:
                return 0.0
            response = self.client.get_balance(target, commitment=Confirmed)
            lamports = int(response.value)
            return lamports / 1e9
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_token_balance(self, token_account: Pubkey) -> float:
        try:
            response = self.client.get_token_account_balance(token_account)
            if response.value:
                return float(response.value.ui_amount or 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get token balance: {e}")
            return 0.0

    def get_token_accounts(self, owner: Optional[Pubkey] = None) -> List[Dict[str, Any]]:
        try:
            target = owner or self.get_pubkey()
            if not target:
                return []

            token_program = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
            opts = TokenAccountOpts(program_id=token_program)
            response = self.client.get_token_accounts_by_owner(target, opts)

            accounts: List[Dict[str, Any]] = []
            for account in response.value:
                accounts.append(
                    {
                        "pubkey": str(account.pubkey),
                        "data": account.account.data,
                    }
                )
            return accounts
        except Exception as e:
            logger.error(f"Failed to get token accounts: {e}")
            return []

    def send_transaction(
        self,
        transaction: Transaction,
        signers: Optional[List[Keypair]] = None,
        priority_fee: int = 0,
    ) -> TransactionResult:
        if Config.is_simulation():
            logger.info(f"[SIMULATION] Would send transaction with priority fee: {priority_fee}")
            return TransactionResult(
                success=True,
                signature="SIMULATED_TX_" + str(int(time.time() * 1000)),
                simulated=True,
            )

        if not self.keypair:
            return TransactionResult(success=False, error="No keypair available for signing")

        all_signers: List[Keypair] = [self.keypair] + (signers or [])

        # FIX: solders.transaction.Transaction doesn't expose with_recent_blockhash in type stubs
        # and also doesn't allow setting .recent_blockhash. Use the Solana RPC client's
        # send_transaction flow without manually mutating the transaction blockhash.
        #
        # In solana-py, send_transaction will compile/sign and fetch a recent blockhash internally.
        tx_opts = TxOpts(skip_preflight=False, preflight_commitment=Confirmed)

        for attempt in range(int(getattr(Config, "TX_RETRY_COUNT", 1))):
            try:
                response = self.client.send_transaction(transaction, *all_signers, opts=tx_opts)
                signature = str(response.value)
                logger.info(f"Transaction sent: {signature}")
                return TransactionResult(success=True, signature=signature)

            except Exception as e:
                logger.warning(f"Transaction attempt {attempt + 1} failed: {e}")
                if attempt < int(getattr(Config, "TX_RETRY_COUNT", 1)) - 1:
                    time.sleep(float(getattr(Config, "TX_RETRY_DELAY_MS", 0)) / 1000.0)

        return TransactionResult(success=False, error="Transaction failed after all retries")

    def confirm_transaction(self, signature: str, timeout: int = 30) -> bool:
        if Config.is_simulation():
            return True

        try:
            sig = Signature.from_string(signature)
            start = time.time()
            while time.time() - start < float(timeout):
                response = self.client.get_signature_statuses([sig])
                if response.value and response.value[0]:
                    status = response.value[0]
                    if status.confirmation_status:
                        logger.info(f"Transaction confirmed: {signature}")
                        return True
                time.sleep(0.5)
            return False
        except Exception as e:
            logger.error(f"Failed to confirm transaction: {e}")
            return False

    def get_recent_blockhash(self) -> Optional[str]:
        try:
            response = self.client.get_latest_blockhash()
            return str(response.value.blockhash)
        except Exception as e:
            logger.error(f"Failed to get recent blockhash: {e}")
            return None

    def get_slot(self) -> int:
        try:
            return int(self.client.get_slot().value)
        except Exception as e:
            logger.error(f"Failed to get slot: {e}")
            return 0

