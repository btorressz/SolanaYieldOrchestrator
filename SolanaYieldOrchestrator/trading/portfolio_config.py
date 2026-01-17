from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
import json

from config import Config
from infra.redis_client import RedisCache
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_memory_configs: Dict[str, "PortfolioConfig"] = {}


@dataclass
class PortfolioConfig:
    initial_nav: float = 10000.0
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "basis": 0.5,
        "funding": 0.3,
        "cash": 0.2,
    })
    asset_weights: Dict[str, float] = field(default_factory=lambda: {
        "SOL": 0.6,
        "mSOL": 0.2,
        "BTC": 0.1,
        "ETH": 0.1,
    })
    enabled_assets: list = field(default_factory=lambda: ["SOL", "BTC", "ETH", "mSOL"])
    
    def __post_init__(self):
        self.normalize_weights()
        self._validate_enabled_assets()
    
    def normalize_weights(self):
        self.strategy_weights = self._normalize_dict(self.strategy_weights)
        self.asset_weights = self._normalize_dict(self.asset_weights)
    
    def _validate_enabled_assets(self):
        supported = set(Config.SUPPORTED_ASSETS)
        self.enabled_assets = [a for a in self.enabled_assets if a in supported]
        if not self.enabled_assets:
            self.enabled_assets = ["SOL", "BTC", "ETH"]
        
        for asset in list(self.asset_weights.keys()):
            if asset not in self.enabled_assets:
                del self.asset_weights[asset]
        
        for asset in self.enabled_assets:
            if asset not in self.asset_weights:
                self.asset_weights[asset] = 0.0
        
        self.asset_weights = self._normalize_dict(self.asset_weights)
    
    @staticmethod
    def _normalize_dict(weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            return weights
        
        total = sum(weights.values())
        if total <= 0:
            return weights
        
        if abs(total - 1.0) > 0.0001:
            return {k: v / total for k, v in weights.items()}
        
        return weights
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_nav": self.initial_nav,
            "strategy_weights": self.strategy_weights,
            "asset_weights": self.asset_weights,
            "enabled_assets": self.enabled_assets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioConfig":
        return cls(
            initial_nav=data.get("initial_nav", 10000.0),
            strategy_weights=data.get("strategy_weights", {
                "basis": 0.5,
                "funding": 0.3,
                "cash": 0.2,
            }),
            asset_weights=data.get("asset_weights", {
                "SOL": 0.6,
                "mSOL": 0.2,
                "BTC": 0.1,
                "ETH": 0.1,
            }),
            enabled_assets=data.get("enabled_assets", ["SOL", "BTC", "ETH", "mSOL"]),
        )
    
    @classmethod
    def from_profile(cls, profile_name: str) -> "PortfolioConfig":
        profile = Config.get_strategy_profile(profile_name)
        allocations = profile.get("allocations", {})
        
        strategy_weights = {
            "basis": allocations.get("basis_harvester", 0.4),
            "funding": allocations.get("funding_rotator", 0.35),
            "cash": allocations.get("cash", 0.25),
        }
        
        return cls(
            initial_nav=10000.0,
            strategy_weights=strategy_weights,
        )
    
    def validate(self) -> tuple[bool, Optional[str]]:
        if self.initial_nav <= 0:
            return False, "initial_nav must be greater than 0"
        
        if self.initial_nav > 100000000:
            return False, "initial_nav cannot exceed 100,000,000"
        
        strategy_sum = sum(self.strategy_weights.values())
        if abs(strategy_sum - 1.0) > 0.01:
            return False, f"strategy_weights must sum to 1.0 (got {strategy_sum:.2f})"
        
        if any(v < 0 for v in self.strategy_weights.values()):
            return False, "strategy_weights values cannot be negative"
        
        if self.asset_weights:
            asset_sum = sum(self.asset_weights.values())
            if abs(asset_sum - 1.0) > 0.01:
                return False, f"asset_weights must sum to 1.0 (got {asset_sum:.2f})"
            
            if any(v < 0 for v in self.asset_weights.values()):
                return False, "asset_weights values cannot be negative"
        
        return True, None


def get_portfolio_config(session_id: str) -> PortfolioConfig:
    global _memory_configs
    
    cached = RedisCache.get_portfolio_config(session_id)
    if cached:
        return PortfolioConfig.from_dict(cached)
    
    if session_id in _memory_configs:
        return _memory_configs[session_id]
    
    return PortfolioConfig()


def set_portfolio_config(session_id: str, config: PortfolioConfig) -> bool:
    global _memory_configs
    
    config.normalize_weights()
    
    valid, error = config.validate()
    if not valid:
        logger.warning(f"Invalid portfolio config for session {session_id}: {error}")
        return False
    
    _memory_configs[session_id] = config
    
    RedisCache.set_portfolio_config(session_id, config.to_dict())
    
    logger.info(f"Portfolio config updated for session {session_id}: NAV=${config.initial_nav:.2f}")
    return True


def update_portfolio_config(
    session_id: str,
    initial_nav: Optional[float] = None,
    strategy_weights: Optional[Dict[str, float]] = None,
    asset_weights: Optional[Dict[str, float]] = None,
    enabled_assets: Optional[list] = None,
) -> tuple[PortfolioConfig, Optional[str]]:
    current = get_portfolio_config(session_id)
    
    new_config = PortfolioConfig(
        initial_nav=initial_nav if initial_nav is not None else current.initial_nav,
        strategy_weights=strategy_weights if strategy_weights is not None else current.strategy_weights.copy(),
        asset_weights=asset_weights if asset_weights is not None else current.asset_weights.copy(),
        enabled_assets=enabled_assets if enabled_assets is not None else current.enabled_assets.copy(),
    )
    
    valid, error = new_config.validate()
    if not valid:
        return current, error
    
    set_portfolio_config(session_id, new_config)
    return new_config, None


def delete_portfolio_config(session_id: str):
    global _memory_configs
    
    if session_id in _memory_configs:
        del _memory_configs[session_id]
    
    RedisCache.delete(f"portfolio:{session_id}")


def get_default_weights_from_profile(profile_name: str) -> Dict[str, float]:
    profile = Config.get_strategy_profile(profile_name)
    allocations = profile.get("allocations", {})
    
    return {
        "basis": allocations.get("basis_harvester", 0.4),
        "funding": allocations.get("funding_rotator", 0.35),
        "cash": allocations.get("cash", 0.25),
    }
