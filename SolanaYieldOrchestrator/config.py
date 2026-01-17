import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODE = os.getenv("MODE", "simulation")
    
    MARKET_DATA_MODE = os.getenv("MARKET_DATA_MODE", "auto")  # "auto" | "live" | "mock"
    API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "3.0"))
    
    SUPPORTED_ASSETS = ["SOL", "BTC", "ETH", "mSOL", "BONK", "XRP", "USDC", "USDT", "JTO", "JUP", "ORCA", "RAY"]
    
    ASSET_MINT_ADDRESSES = {
        "SOL": "So11111111111111111111111111111111111111112",
        "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",
        "ETH": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
        "mSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "XRP": "7RMr7KGzPF6EjWL7DTU9jR5gN6WqJMJxJ7qMGPeQGXQd",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "JTO": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
        "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
        "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    }
    
    ASSET_COINGECKO_IDS = {
        "SOL": "solana",
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "mSOL": "msol",
        "BONK": "bonk",
        "XRP": "ripple",
        "USDC": "usd-coin",
        "USDT": "tether",
        "JTO": "jito-governance-token",
        "JUP": "jupiter-exchange-solana",
        "ORCA": "orca",
        "RAY": "raydium",
    }
    
    ASSET_KRAKEN_PAIRS = {
        "SOL": "SOLUSD",
        "BTC": "XBTUSD",
        "ETH": "ETHUSD",
        "XRP": "XRPUSD",
    }
    
    ASSET_HL_SYMBOLS = {
        "SOL": "SOL",
        "BTC": "BTC",
        "ETH": "ETH",
    }
    
    ASSET_DRIFT_SYMBOLS = {
        "SOL": "SOL-PERP",
        "BTC": "BTC-PERP",
        "ETH": "ETH-PERP",
    }
    
    ASSET_TYPES = {
        "SOL": "native",
        "BTC": "wrapped",
        "ETH": "wrapped",
        "mSOL": "lst",
        "BONK": "meme",
        "XRP": "wrapped",
        "USDC": "stablecoin",
        "USDT": "stablecoin",
        "JTO": "governance",
        "JUP": "governance",
        "ORCA": "dex",
        "RAY": "dex",
    }
    
    @classmethod
    def get_asset_info(cls, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "mint_address": cls.ASSET_MINT_ADDRESSES.get(symbol, ""),
            "type": cls.ASSET_TYPES.get(symbol, "unknown"),
            "decimals": cls.ASSET_DECIMALS.get(symbol, 9),
            "coingecko_id": cls.ASSET_COINGECKO_IDS.get(symbol),
            "kraken_pair": cls.ASSET_KRAKEN_PAIRS.get(symbol),
            "hyperliquid_symbol": cls.ASSET_HL_SYMBOLS.get(symbol),
            "has_hyperliquid": symbol in cls.ASSET_HL_SYMBOLS,
            "drift_symbol": cls.ASSET_DRIFT_SYMBOLS.get(symbol),
            "has_drift": symbol in cls.ASSET_DRIFT_SYMBOLS,
            "is_stablecoin": cls.ASSET_TYPES.get(symbol) == "stablecoin",
        }
    
    @classmethod
    def get_all_assets_info(cls) -> list:
        return [cls.get_asset_info(s) for s in cls.SUPPORTED_ASSETS]
    
    ASSET_DECIMALS = {
        "SOL": 9, "mSOL": 9, "BONK": 5,
        "BTC": 8, "ETH": 8,
        "XRP": 6, "USDC": 6, "USDT": 6,
        "JTO": 9, "JUP": 6, "ORCA": 6, "RAY": 6,
    }
    
    @classmethod
    def get_mint_address(cls, symbol: str) -> str:
        return cls.ASSET_MINT_ADDRESSES.get(symbol, "")
    
    REDIS_URL = os.getenv("REDIS_URL", "")
    REDIS_PRICE_TTL_SECONDS = int(os.getenv("REDIS_PRICE_TTL_SECONDS", "10"))
    REDIS_ROUTE_TTL_SECONDS = int(os.getenv("REDIS_ROUTE_TTL_SECONDS", "60"))
    REDIS_PERP_TTL_SECONDS = int(os.getenv("REDIS_PERP_TTL_SECONDS", "15"))
    REDIS_METRICS_TTL_SECONDS = int(os.getenv("REDIS_METRICS_TTL_SECONDS", "300"))
    
    SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    SOLANA_WS_URL = os.getenv("SOLANA_WS_URL", "wss://api.mainnet-beta.solana.com")
    SOLANA_PRIVATE_KEY = os.getenv("SOLANA_PRIVATE_KEY", "")
    SOLANA_KEYPAIR_PATH = os.getenv("SOLANA_KEYPAIR_PATH", "")
    
    HYPERLIQUID_ENABLED = os.getenv("HYPERLIQUID_ENABLED", "false").lower() == "true"
    HYPERLIQUID_API_URL = os.getenv("HYPERLIQUID_API_URL", "https://api.hyperliquid.xyz")
    HYPERLIQUID_WS_URL = os.getenv("HYPERLIQUID_WS_URL", "wss://api.hyperliquid.xyz/ws")
    HYPERLIQUID_PRIVATE_KEY = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    HYPERLIQUID_TESTNET = os.getenv("HYPERLIQUID_TESTNET", "false").lower() == "true"
    
    PYTH_ENABLED = os.getenv("PYTH_ENABLED", "true").lower() == "true"
    PYTH_API_URL = os.getenv("PYTH_API_URL", "https://hermes.pyth.network/api/latest_price_feeds")
    MAX_ORACLE_DEVIATION_BPS = int(os.getenv("MAX_ORACLE_DEVIATION_BPS", "100"))
    
    AGENT_KIT_ENABLED = os.getenv("AGENT_KIT_ENABLED", "false").lower() == "true"
    AGENT_KIT_LIVE_TRADING = os.getenv("AGENT_KIT_LIVE_TRADING", "false").lower() == "true"
    
    DYNAMIC_PRIORITY_FEES = os.getenv("DYNAMIC_PRIORITY_FEES", "false").lower() == "true"
    
    RATE_LIMIT_JUPITER_RPS = int(os.getenv("RATE_LIMIT_JUPITER_RPS", "10"))
    RATE_LIMIT_JUPITER_RPM = int(os.getenv("RATE_LIMIT_JUPITER_RPM", "300"))
    RATE_LIMIT_DRIFT_RPS = int(os.getenv("RATE_LIMIT_DRIFT_RPS", "10"))
    RATE_LIMIT_DRIFT_RPM = int(os.getenv("RATE_LIMIT_DRIFT_RPM", "300"))
    RATE_LIMIT_HYPERLIQUID_RPS = int(os.getenv("RATE_LIMIT_HYPERLIQUID_RPS", "10"))
    RATE_LIMIT_HYPERLIQUID_RPM = int(os.getenv("RATE_LIMIT_HYPERLIQUID_RPM", "300"))
    
    DRIFT_ENV = os.getenv("DRIFT_ENV", "mainnet")
    DRIFT_SUBACCOUNT_ID = int(os.getenv("DRIFT_SUBACCOUNT_ID", "0"))
    
    JUPITER_API_URL = os.getenv("JUPITER_API_URL", "https://quote-api.jup.ag/v6")
    JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")
    
    COINGECKO_API_URL = os.getenv("COINGECKO_API_URL", "https://api.coingecko.com/api/v3")
    COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
    
    KRAKEN_API_URL = os.getenv("KRAKEN_API_URL", "https://api.kraken.com/0/public")
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")
    
    BASIS_ENTRY_THRESHOLD_BPS = int(os.getenv("BASIS_ENTRY_THRESHOLD_BPS", "50"))
    BASIS_EXIT_THRESHOLD_BPS = int(os.getenv("BASIS_EXIT_THRESHOLD_BPS", "10"))
    FUNDING_MIN_APY_THRESHOLD = float(os.getenv("FUNDING_MIN_APY_THRESHOLD", "5.0"))
    FUNDING_TOP_N_MARKETS = int(os.getenv("FUNDING_TOP_N_MARKETS", "3"))
    ROTATION_EPOCH_SECONDS = int(os.getenv("ROTATION_EPOCH_SECONDS", "3600"))
    
    ALLOCATION_BASIS_HARVESTER = float(os.getenv("ALLOCATION_BASIS_HARVESTER", "0.50"))
    ALLOCATION_FUNDING_ROTATOR = float(os.getenv("ALLOCATION_FUNDING_ROTATOR", "0.30"))
    ALLOCATION_CASH = float(os.getenv("ALLOCATION_CASH", "0.20"))
    
    MAX_POSITION_SIZE_USD = float(os.getenv("MAX_POSITION_SIZE_USD", "10000"))
    MAX_SLIPPAGE_BPS = int(os.getenv("MAX_SLIPPAGE_BPS", "100"))
    MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "10.0"))
    
    TX_RETRY_COUNT = int(os.getenv("TX_RETRY_COUNT", "3"))
    TX_RETRY_DELAY_MS = int(os.getenv("TX_RETRY_DELAY_MS", "500"))
    
    PRIORITY_FEE_CHEAP = int(os.getenv("PRIORITY_FEE_CHEAP", "1000"))
    PRIORITY_FEE_BALANCED = int(os.getenv("PRIORITY_FEE_BALANCED", "10000"))
    PRIORITY_FEE_FAST = int(os.getenv("PRIORITY_FEE_FAST", "100000"))
    
    SECRET_KEY = os.getenv("SESSION_SECRET", os.urandom(24).hex())
    
    STRATEGY_PROFILES = {
        "conservative": {
            "name": "Conservative",
            "description": "Low risk, focus on stable yield with minimal drawdown",
            "strategies": {
                "basis": True,
                "funding": True,
                "carry": False,
                "volatility": False,
                "basket": False
            },
            "risk_limits": {
                "max_position_size_usd": 5000.0,
                "max_leverage": 2.0,
                "max_drawdown_pct": 5.0,
                "max_slippage_bps": 50
            },
            "thresholds": {
                "basis_entry_bps": 75,
                "basis_exit_bps": 15,
                "funding_min_apy": 10.0,
                "funding_top_n": 2
            },
            "allocations": {
                "basis_harvester": 0.30,
                "funding_rotator": 0.20,
                "cash": 0.50
            }
        },
        "balanced": {
            "name": "Balanced",
            "description": "Moderate risk with balanced yield and risk exposure",
            "strategies": {
                "basis": True,
                "funding": True,
                "carry": True,
                "volatility": False,
                "basket": False
            },
            "risk_limits": {
                "max_position_size_usd": 10000.0,
                "max_leverage": 5.0,
                "max_drawdown_pct": 10.0,
                "max_slippage_bps": 100
            },
            "thresholds": {
                "basis_entry_bps": 50,
                "basis_exit_bps": 10,
                "funding_min_apy": 5.0,
                "funding_top_n": 3
            },
            "allocations": {
                "basis_harvester": 0.40,
                "funding_rotator": 0.35,
                "cash": 0.25
            }
        },
        "aggro": {
            "name": "Aggressive Yield",
            "description": "High risk, maximum yield pursuit with higher leverage",
            "strategies": {
                "basis": True,
                "funding": True,
                "carry": True,
                "volatility": True,
                "basket": True
            },
            "risk_limits": {
                "max_position_size_usd": 25000.0,
                "max_leverage": 10.0,
                "max_drawdown_pct": 20.0,
                "max_slippage_bps": 200
            },
            "thresholds": {
                "basis_entry_bps": 25,
                "basis_exit_bps": 5,
                "funding_min_apy": 2.0,
                "funding_top_n": 5
            },
            "allocations": {
                "basis_harvester": 0.45,
                "funding_rotator": 0.45,
                "cash": 0.10
            }
        }
    }
    
    @classmethod
    def is_simulation(cls) -> bool:
        return cls.MODE.lower() == "simulation"
    
    @classmethod
    def get_priority_fee(cls, profile: str) -> int:
        profiles = {
            "cheap": cls.PRIORITY_FEE_CHEAP,
            "balanced": cls.PRIORITY_FEE_BALANCED,
            "fast": cls.PRIORITY_FEE_FAST
        }
        return profiles.get(profile.lower(), cls.PRIORITY_FEE_BALANCED)
    
    @classmethod
    def get_allocations(cls) -> dict:
        return {
            "basis_harvester": cls.ALLOCATION_BASIS_HARVESTER,
            "funding_rotator": cls.ALLOCATION_FUNDING_ROTATOR,
            "cash": cls.ALLOCATION_CASH
        }
    
    @classmethod
    def get_strategy_profile(cls, profile_name: str) -> dict:
        """Get a strategy profile by name. Returns balanced if not found."""
        return cls.STRATEGY_PROFILES.get(profile_name.lower(), cls.STRATEGY_PROFILES["balanced"])
    
    @classmethod
    def get_all_profiles(cls) -> dict:
        """Return all available strategy profiles."""
        return {
            name: {
                "name": profile["name"],
                "description": profile["description"]
            }
            for name, profile in cls.STRATEGY_PROFILES.items()
        }
