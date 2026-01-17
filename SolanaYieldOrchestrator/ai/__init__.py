"""
AI Agent Bridge Module for Solana Yield Orchestrator

This package provides a thin wrapper around orchestrator functionality
for integration with Solana Agent Kit. All operations respect existing
risk limits and are restricted to simulation mode by default.

IMPORTANT: This is Phase 2 / experimental functionality.
Enable with AGENT_KIT_ENABLED=true environment variable.
"""

from .agent_bridge import AgentBridge, AgentTool

__all__ = ['AgentBridge', 'AgentTool']
