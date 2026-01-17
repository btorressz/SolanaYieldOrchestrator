 # Solana Yield Orchestrator

A professional-grade, full-stack Python Flask application for orchestrating advanced DeFi yield strategies on the Solana blockchain. This platform provides institutional-level analytics, multi-venue trading capabilities, real-time monitoring, and comprehensive risk management for yield farming operations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [File Documentation](#file-documentation)
- [Architecture](#architecture)
- [Supported Assets](#supported-assets)
- [Multi-Venue Integration](#multi-venue-integration)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Installation & Usage](#installation--usage)
- [Technology Stack](#technology-stack)

---

## Project Overview

The Solana Yield Orchestrator is an automated trading and analytics system designed for DeFi yield optimization on the Solana blockchain. It combines multiple yield-generating strategies with institutional-grade features:

- **Yield Strategy Execution**: Basis harvesting, funding rate rotation, cross-venue arbitrage
- **Multi-Venue Trading**: Unified interface across Drift Protocol, Jupiter Aggregator, Hyperliquid
- **Real-Time Analytics**: 17+ dashboard panels with live charts, PnL tracking, risk metrics
- **Oracle Quality Monitoring**: Pyth Network integration for trade protection
- **MEV Protection**: Route analysis for safer trading
- **Paper Trading**: Full simulation environment for strategy testing
- **Per-Venue Price Breakdown**: Visibility into 6 data sources (Jupiter, CoinGecko, Kraken, Hyperliquid, Drift, Pyth)
- **AI Agent Integration**: Experimental Solana Agent Kit for automated execution (Phase 2, disabled by default)

**Operating Modes:**
- **Simulation Mode** (default): Paper trading with mock or live data
- **Live Mode**: Real trading with actual capital

  
## Key Features

### 1. Multi-Source Price Aggregation (6 Venues)

| Source | Badge | Purpose | Coverage |
|--------|-------|---------|----------|
| **Jupiter Aggregator** | `J` | On-chain DEX aggregation | All Solana tokens |
| **CoinGecko** | `CG` | Reference price validation | All tokens |
| **Kraken** | `K` | CEX price cross-reference | SOL, BTC, ETH, XRP |
| **Hyperliquid** | `HL` | Perpetual venue prices | SOL, BTC, ETH |
| **Drift Protocol** | `DR` | Perpetual mark prices | SOL-PERP, BTC-PERP, ETH-PERP |
| **Pyth Network** | `PY` | Decentralized oracle feeds | All supported tokens |

### 2. Pyth Oracle Integration

- **Status Indicators**: Clean (≤25 bps), Watch (25-100 bps), Flagged (>100 bps)
- **Automatic Trade Blocking**: Prevents trades when deviation exceeds threshold
- **Configurable Threshold**: `MAX_ORACLE_DEVIATION_BPS` environment variable
- **Real-Time Monitoring**: Inline display in Live Prices panel

### 3. Cross-Venue Analytics

- **Basis Comparison**: Spot vs perpetual price spreads across venues
- **Funding Rate Analysis**: Side-by-side venue funding rates
- **MEV Risk Assessment**: Classifies DEX routes (Low/Medium/High risk)
- **Venue Health Monitoring**: Circuit breaker status and latency tracking

### 4. Dynamic Asset Universe

**12 Supported Assets:**
- **Native**: SOL
- **Wrapped**: BTC, ETH, XRP
- **Liquid Staking**: mSOL
- **Governance**: JTO, JUP
- **DEX Tokens**: ORCA, RAY
- **Stablecoins**: USDC, USDT
- **Memes**: BONK

All UI components dynamically adapt to enabled assets with real-time updates.

---


## Project Structure

```
solana-yield-orchestrator/
├── app.py                          # Main Flask application (2433 lines)
├── config.py                       # Configuration management (301 lines)
├── simulator.py                    # Monte Carlo simulation engine (10189 lines)
├── requirements.txt                # Python dependencies
│
├── ai/                             # AI Agent Integration (Phase 2)
│   ├── __init__.py                 # Package exports
│   └── agent_bridge.py             # Agent tool definitions and execution
│
├── infra/                          # Infrastructure Layer (blockchain & venues)
│   ├── __init__.py
│   ├── solana_client.py            # Solana RPC client wrapper
│   ├── jupiter_client.py           # Jupiter aggregator integration
│   ├── drift_client.py             # Drift Protocol perpetual trading
│   ├── hyperliquid_client.py       # Hyperliquid perpetual venue (optional)
│   ├── pyth_client.py              # Pyth oracle price feeds
│   ├── perp_venue.py               # Abstract perpetual venue interface
│   ├── solana_chain_monitor.py     # Chain state monitoring
│   ├── redis_client.py             # Redis caching layer
│   ├── rate_limiter.py             # API rate limiting
│   ├── priority_router.py          # Multi-venue execution routing
│   └── metrics_tracker.py          # HFT metrics collection
│
├── data/                           # Data Layer (aggregation & analytics)
│   ├── __init__.py
│   ├── data_fetcher.py             # Multi-source price aggregation
│   ├── analytics.py                # Performance analytics
│   └── mock_data.py                # Mock data for offline testing
│
├── strategies/                     # Yield Strategy Implementations
│   ├── __init__.py
│   ├── basis_harvester.py          # Spot-perp basis trading
│   ├── funding_rotator.py          # Funding rate arbitrage
│   ├── perp_spread.py              # Cross-perp spread trading
│   ├── cross_venue_funding_arb.py  # Multi-venue funding arb
│   ├── hedged_basket.py            # Hedged portfolio basket
│   ├── volatility_scaler.py        # Vol-scaled position sizing
│   └── carry_optimizer.py          # Carry trade optimization
│
├── trading/                        # Trading Execution Layer
│   ├── __init__.py
│   ├── paper_account.py            # Simulation paper trading (701 lines)
│   └── portfolio_config.py         # Portfolio configuration (207 lines)
│
├── vault/                          # Risk & Portfolio Management
│   ├── __init__.py
│   ├── vault_manager.py            # Risk limits and position sizing
│   └── mock_hedge_basket.py        # Mock hedge implementation
│
├── utils/                          # Utility Functions
│   ├── __init__.py
│   ├── logging_utils.py            # Centralized logging
│   └── risk_limits.py              # Risk limit definitions
│
├── templates/                      # HTML Templates
│   └── index.html                  # Main dashboard (899 lines, 17+ panels)
│
├── static/                         # Frontend Assets
│   ├── main.js                     # Dashboard logic (2115 lines)
│   └── style.css                   # Styling (2448 lines)
│
└── .replit & .gitignore           # Project configuration
```

---

## File Documentation

### Core Application Files

#### `app.py` (Main Flask Server - 2433 lines)
**Purpose:** Central Flask REST API server orchestrating all system components
**Key Responsibilities:**
- Flask app initialization and CORS setup
- Client initialization (Solana, Jupiter, Drift, Hyperliquid, Pyth)
- 60+ REST API endpoints for dashboard, portfolio, trading, streaming
- Server-Sent Events (SSE) for real-time updates
- Paper trading execution and simulation
- Risk management and position monitoring

**Key Sections:**
- **Initialization**: Client setup for blockchain, venues, data sources
- **Portfolio API**: `/api/portfolio/*` endpoints for position and state management
- **Pricing API**: `/api/prices`, `/api/venue/*` for price aggregation
- **Trading API**: `/api/trade/*` for order execution and simulation
- **Analytics API**: `/api/metrics`, `/api/hft/*` for performance tracking
- **Agent API**: `/api/agent/*` for experimental AI integration (when enabled)
- **Streaming**: `/events` endpoint for live dashboard updates

#### `config.py` (Configuration Manager - 301 lines)
**Purpose:** Centralized configuration management with feature flags and environment variables
**Key Components:**
- **Asset Configuration**: 12 supported assets with mint addresses, decimals, exchange mappings
- **Venue Credentials**: Solana RPC, Drift, Hyperliquid, Pyth endpoints
- **Feature Flags**: Hyperliquid, Pyth, Agent Kit enable/disable
- **Strategy Parameters**: Basis thresholds, funding rates, rotation epochs
- **Risk Limits**: Max position size, max leverage, max drawdown, slippage
- **Rate Limiting**: Per-venue API rate limits (RPS and RPM)
- **Trading Execution**: Priority fees, retry logic, TWAP parameters

**Key Methods:**
- `get_asset_info()`: Returns comprehensive asset metadata
- `get_all_assets_info()`: Lists all configured assets
- `get_mint_address()`: Maps symbol to SPL token mint
- `is_simulation()`: Mode detection

#### `simulator.py` (Monte Carlo Simulation Engine - 10189 lines)
**Purpose:** Paper trading engine with Monte Carlo simulation for strategy backtesting
**Key Features:**
- Monte Carlo price path generation
- Position lifecycle management (entry, hold, exit)
- PnL tracking and performance metrics
- Sharpe ratio, Sortino ratio, volatility, max drawdown calculation
- Risk limit enforcement
- Strategy execution in simulated environment

---


---
