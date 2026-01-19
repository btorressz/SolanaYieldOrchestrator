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

## Folders

### `ai/` - AI Agent Integration (Experimental)
Contains the Solana Agent Kit integration for automated strategy execution.

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `agent_bridge.py` | 8 agent tools: 5 read-only (portfolio state, metrics, prices, funding, oracle) + 3 safe actions (simulation, config update, scenario testing) |

**Safety:** All agent actions require simulation mode and respect existing risk limits.

---

### `infra/` - Infrastructure Layer
Low-level blockchain and exchange integrations.

| File | Lines | Purpose |
|------|-------|---------|
| `solana_client.py` | 205 | Solana RPC wrapper for balance queries and transactions |
| `jupiter_client.py` | 1,008 | Jupiter DEX aggregator for swap quotes and MEV analysis |
| `drift_client.py` | - | Drift Protocol perpetual trading and funding rates |
| `hyperliquid_client.py` | - | Optional second perp venue (enable with HYPERLIQUID_ENABLED=true) |
| `pyth_client.py` | 375 | Pyth oracle integration with v2 API for live price feeds |
| `perp_venue.py` | - | Abstract interface for unified perp trading across venues |
| `solana_chain_monitor.py` | 423 | Real-time chain state monitoring |
| `redis_client.py` | 305 | High-performance caching with category-specific TTLs |
| `rate_limiter.py` | 158 | Per-venue API rate limiting (RPS and RPM) |
| `priority_router.py` | - | Multi-venue execution routing and TWAP |
| `metrics_tracker.py` | - | HFT metrics (latency, API calls, execution quality) |

---

### `data/` - Data Layer
Price aggregation and analytics.

| File | Purpose |
|------|---------|
| `data_fetcher.py` | Aggregates prices from 6 venues (Jupiter, CoinGecko, Kraken, Hyperliquid, Drift, Pyth) with weighted averaging and outlier detection |
| `analytics.py` | Calculates Sharpe ratio, Sortino ratio, volatility, max drawdown |
| `mock_data.py` | Generates realistic mock data for offline development |

---

### `strategies/` - Yield Strategy Implementations
7 automated trading strategies.

| Strategy | File | Description |
|----------|------|-------------|
| Basis Harvester | `basis_harvester.py` | Trades spot vs perpetual price spreads |
| Funding Rotator | `funding_rotator.py` | Rotates positions to highest funding rate markets |
| Perp Spread | `perp_spread.py` | Cross-perpetual spread trading |
| Cross-Venue Arb | `cross_venue_funding_arb.py` | Funding arbitrage between Drift and Hyperliquid |
| Hedged Basket | `hedged_basket.py` | Maintains correlated asset baskets with hedging |
| Volatility Scaler | `volatility_scaler.py` | Adjusts position sizes based on volatility |
| Carry Optimizer | `carry_optimizer.py` | Optimizes carry trade opportunities |

---

### `trading/` - Trading Execution Layer
Paper trading and portfolio management.

| File | Lines | Purpose |
|------|-------|---------|
| `paper_account.py` | 701 | Simulates trading without real capital - tracks positions, PnL, and metrics |
| `portfolio_config.py` | 207 | Manages strategy allocations and profiles (Aggressive, Balanced, Conservative) |

---

### `vault/` - Risk Management
Portfolio constraints and position sizing.

| File | Lines | Purpose |
|------|-------|---------|
| `vault_manager.py` | 388 | Enforces risk limits (max position, leverage, drawdown, slippage) |
| `mock_hedge_basket.py` | 459 | Simulates hedging strategies for portfolio protection |

---

### `utils/` - Utility Functions

| File | Purpose |
|------|---------|
| `logging_utils.py` | Centralized logging with color-coded output |
| `risk_limits.py` | Risk limit type definitions and validation |

---

### `templates/` - HTML Templates

| File | Lines | Purpose |
|------|-------|---------|
| `index.html` | 899 | Main dashboard with 17+ panels: Portfolio Overview, Live Prices, Positions, Funding Rates, Risk Metrics, PnL Breakdown, Chain State, and more |

---
