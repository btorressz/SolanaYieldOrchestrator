from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskCheck:
    passed: bool
    level: RiskLevel
    message: str
    metric_name: str
    current_value: float
    limit_value: float

class RiskLimits:
    def __init__(self):
        self.max_position_size_usd = Config.MAX_POSITION_SIZE_USD
        self.max_slippage_bps = Config.MAX_SLIPPAGE_BPS
        self.max_drawdown_pct = Config.MAX_DRAWDOWN_PCT
        
        self.max_single_trade_pct = 10.0
        self.max_leverage = 5.0
        self.min_cash_buffer_pct = 5.0
        self.max_concentration_pct = 40.0
        
        self._peak_nav = 0.0
        self._current_drawdown = 0.0
    
    def check_position_size(self, size_usd: float) -> RiskCheck:
        passed = size_usd <= self.max_position_size_usd
        
        if size_usd <= self.max_position_size_usd * 0.5:
            level = RiskLevel.LOW
        elif size_usd <= self.max_position_size_usd * 0.8:
            level = RiskLevel.MEDIUM
        elif passed:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        return RiskCheck(
            passed=passed,
            level=level,
            message=f"Position size ${size_usd:.2f} vs limit ${self.max_position_size_usd:.2f}",
            metric_name="position_size",
            current_value=size_usd,
            limit_value=self.max_position_size_usd
        )
    
    def check_slippage(self, slippage_bps: float) -> RiskCheck:
        passed = slippage_bps <= self.max_slippage_bps
        
        if slippage_bps <= self.max_slippage_bps * 0.5:
            level = RiskLevel.LOW
        elif slippage_bps <= self.max_slippage_bps * 0.8:
            level = RiskLevel.MEDIUM
        elif passed:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        return RiskCheck(
            passed=passed,
            level=level,
            message=f"Slippage {slippage_bps:.1f} bps vs limit {self.max_slippage_bps} bps",
            metric_name="slippage",
            current_value=slippage_bps,
            limit_value=float(self.max_slippage_bps)
        )
    
    def check_drawdown(self, current_nav: float) -> RiskCheck:
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav
        
        if self._peak_nav > 0:
            self._current_drawdown = ((self._peak_nav - current_nav) / self._peak_nav) * 100
        else:
            self._current_drawdown = 0.0
        
        passed = self._current_drawdown <= self.max_drawdown_pct
        
        if self._current_drawdown <= self.max_drawdown_pct * 0.3:
            level = RiskLevel.LOW
        elif self._current_drawdown <= self.max_drawdown_pct * 0.6:
            level = RiskLevel.MEDIUM
        elif passed:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        return RiskCheck(
            passed=passed,
            level=level,
            message=f"Drawdown {self._current_drawdown:.2f}% vs limit {self.max_drawdown_pct}%",
            metric_name="drawdown",
            current_value=self._current_drawdown,
            limit_value=self.max_drawdown_pct
        )
    
    def check_leverage(self, total_position_value: float, account_value: float) -> RiskCheck:
        if account_value <= 0:
            return RiskCheck(
                passed=False,
                level=RiskLevel.CRITICAL,
                message="Account value is zero or negative",
                metric_name="leverage",
                current_value=float('inf'),
                limit_value=self.max_leverage
            )
        
        current_leverage = total_position_value / account_value
        passed = current_leverage <= self.max_leverage
        
        if current_leverage <= self.max_leverage * 0.5:
            level = RiskLevel.LOW
        elif current_leverage <= self.max_leverage * 0.8:
            level = RiskLevel.MEDIUM
        elif passed:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        return RiskCheck(
            passed=passed,
            level=level,
            message=f"Leverage {current_leverage:.2f}x vs limit {self.max_leverage}x",
            metric_name="leverage",
            current_value=current_leverage,
            limit_value=self.max_leverage
        )
    
    def check_concentration(self, position_value: float, total_nav: float) -> RiskCheck:
        if total_nav <= 0:
            return RiskCheck(
                passed=False,
                level=RiskLevel.CRITICAL,
                message="Total NAV is zero or negative",
                metric_name="concentration",
                current_value=100.0,
                limit_value=self.max_concentration_pct
            )
        
        concentration_pct = (position_value / total_nav) * 100
        passed = concentration_pct <= self.max_concentration_pct
        
        if concentration_pct <= self.max_concentration_pct * 0.5:
            level = RiskLevel.LOW
        elif concentration_pct <= self.max_concentration_pct * 0.8:
            level = RiskLevel.MEDIUM
        elif passed:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL
        
        return RiskCheck(
            passed=passed,
            level=level,
            message=f"Concentration {concentration_pct:.2f}% vs limit {self.max_concentration_pct}%",
            metric_name="concentration",
            current_value=concentration_pct,
            limit_value=self.max_concentration_pct
        )
    
    def run_all_checks(
        self,
        position_size_usd: float = 0.0,
        slippage_bps: float = 0.0,
        current_nav: float = 0.0,
        total_position_value: float = 0.0,
        account_value: float = 0.0,
        single_position_value: float = 0.0
    ) -> Dict[str, RiskCheck]:
        checks = {}
        
        if position_size_usd > 0:
            checks["position_size"] = self.check_position_size(position_size_usd)
        
        if slippage_bps > 0:
            checks["slippage"] = self.check_slippage(slippage_bps)
        
        if current_nav > 0:
            checks["drawdown"] = self.check_drawdown(current_nav)
        
        if total_position_value > 0 and account_value > 0:
            checks["leverage"] = self.check_leverage(total_position_value, account_value)
        
        if single_position_value > 0 and current_nav > 0:
            checks["concentration"] = self.check_concentration(single_position_value, current_nav)
        
        return checks
    
    def all_checks_passed(self, checks: Dict[str, RiskCheck]) -> bool:
        return all(check.passed for check in checks.values())
    
    def get_failed_checks(self, checks: Dict[str, RiskCheck]) -> List[RiskCheck]:
        return [check for check in checks.values() if not check.passed]
    
    def get_risk_summary(self, checks: Dict[str, RiskCheck]) -> Dict[str, Any]:
        failed = self.get_failed_checks(checks)
        
        max_level = RiskLevel.LOW
        for check in checks.values():
            if check.level.value > max_level.value:
                max_level = check.level
        
        return {
            "overall_risk_level": max_level.value,
            "all_passed": len(failed) == 0,
            "total_checks": len(checks),
            "failed_checks": len(failed),
            "failed_details": [
                {
                    "metric": c.metric_name,
                    "message": c.message,
                    "level": c.level.value
                }
                for c in failed
            ]
        }
    
    def reset_peak_nav(self):
        self._peak_nav = 0.0
        self._current_drawdown = 0.0
