# Solana Yield Orchestrator - Project Explanation

This document provides a detailed explanation of all files and folders in this project.

---

## Root Directory Files

### `app.py` (2,433 lines)
The main Flask application server. This is the heart of the project that:
- Initializes all blockchain clients (Solana, Jupiter, Drift, Hyperliquid, Pyth)
- Provides 60+ REST API endpoints for the dashboard
- Handles real-time streaming via Server-Sent Events (SSE)
- Manages paper trading simulation and position tracking
- Orchestrates strategy execution in the background

### `config.py` (301 lines)
Centralized configuration management:
- Defines all 12 supported assets (SOL, BTC, ETH, mSOL, BONK, XRP, USDC, USDT, JTO, JUP, ORCA, RAY)
- Contains token mint addresses and exchange mappings
- Feature flags (HYPERLIQUID_ENABLED, PYTH_ENABLED, AGENT_KIT_ENABLED)
- Strategy parameters and risk limits
- API endpoints and rate limits

### `simulator.py` (10,189 lines)
Monte Carlo simulation engine for backtesting:
- Generates realistic price paths
- Tracks positions and PnL
- Calculates performance metrics (Sharpe, Sortino, volatility)
- Enables risk-free strategy testing

### `requirements.txt`
Python package dependencies including Flask, requests, redis, websockets, and Solana SDK.

---
