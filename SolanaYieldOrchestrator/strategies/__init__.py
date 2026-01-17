from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Action:
    action_type: str
    params: Dict[str, Any]
    priority: str = "balanced"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "params": self.params,
            "priority": self.priority
        }

class Strategy(ABC):
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def target_allocation(self) -> float:
        pass
    
    @abstractmethod
    def update_state(self, market_snapshot) -> None:
        pass
    
    @abstractmethod
    def desired_actions(self, vault_state: Dict[str, Any]) -> List[Action]:
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass

from .basis_harvester import BasisHarvester
from .funding_rotator import FundingRotator

__all__ = ["Strategy", "Action", "BasisHarvester", "FundingRotator"]
