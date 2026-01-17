import logging
import sys
from typing import Optional
from datetime import datetime

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_loggers = {}

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root_logger.addHandler(file_handler)
    
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger

class SimulationLogger:
    def __init__(self, name: str = "simulation"):
        self.logger = get_logger(name)
        self.events = []
    
    def log_trade(self, action: str, market: str, size: float, price: float, side: str):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "trade",
            "action": action,
            "market": market,
            "size": size,
            "price": price,
            "side": side
        }
        self.events.append(event)
        self.logger.info(f"[SIM TRADE] {action} {side} {size} {market} @ {price}")
    
    def log_position(self, market: str, size: float, entry_price: float, current_price: float, pnl: float):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "position",
            "market": market,
            "size": size,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl": pnl
        }
        self.events.append(event)
        self.logger.info(f"[SIM POS] {market}: size={size}, entry={entry_price}, current={current_price}, pnl={pnl:.2f}")
    
    def log_rebalance(self, strategy: str, old_allocation: float, new_allocation: float):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "rebalance",
            "strategy": strategy,
            "old_allocation": old_allocation,
            "new_allocation": new_allocation
        }
        self.events.append(event)
        self.logger.info(f"[SIM REBALANCE] {strategy}: {old_allocation:.2%} -> {new_allocation:.2%}")
    
    def get_events(self, event_type: Optional[str] = None) -> list:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events.copy()
    
    def clear(self):
        self.events.clear()

setup_logging()
