# Solana Yield Orchestrator

## Overview
This project is a professional-grade, full-stack Python Flask application designed for orchestrating advanced Decentralized Finance (DeFi) yield strategies on the Solana blockchain. It offers institutional-level analytics, multi-venue trading, real-time monitoring, and comprehensive risk management. The platform's core purpose is to enable users to develop, test, and execute complex yield-generating strategies, perform multi-venue perpetual trading across platforms like Drift and Hyperliquid, monitor oracle price quality via Pyth Network, analyze MEV risk in DEX routes, and track real-time positions and PnL. A key ambition is to offer comprehensive paper trading for strategy validation before deploying real capital, establishing itself as a robust tool for sophisticated DeFi participants.

## User Preferences
| Preference | Default | Description |
|:-----------|:--------|:------------|
| Operating Mode | simulation | Safe paper trading by default |
| Market Data Mode | auto | Tries live APIs, falls back to mock |
| Visualization | Chart.js | Interactive charts and graphs |
| Theme | Purple/Teal/Orange | Solana ecosystem gradient theme |
| Paper Trading | enabled | Risk-free testing by default |
| Strategy Profile | Balanced | Moderate risk settings |
| Real-Time Updates | SSE enabled | Live streaming updates |
| Execution Mode | balanced | Standard order execution |
| Hyperliquid | disabled | Enable with HYPERLIQUID_ENABLED=true |
| Pyth Oracle | enabled | Oracle quality monitoring on by default |
| Stablecoin Precision | 4 decimals | For peg monitoring |
| Agent Kit | disabled | Enable with AGENT_KIT_ENABLED=true |

## System Architecture
The application is structured into modular layers: `infra` (blockchain/venue integrations), `data` (aggregation/analytics), `strategies` (yield implementations), `trading` (execution/paper trading), and `vault` (risk management).

**Key Architectural Decisions:**
- **Modularity**: Organized into distinct layers for maintainability and scalability.
- **Graceful Degradation**: Fallback mechanisms (e.g., mock data, read-only modes) for external dependency failures.
- **Configuration-Driven**: All system settings and feature flags are centralized in `config.py`.
- **Feature Flags**: Integrations like Hyperliquid, Pyth, and an experimental AI Agent Kit are toggleable via environment variables.
- **HFT Metrics**: Comprehensive collection and exposure of high-frequency trading metrics for performance analysis.
- **Dynamic Asset Universe**: Frontend components dynamically adapt based on the enabled asset list.
- **Defense-in-Depth Security**: Multiple validation layers for agent execution and trading, especially for the experimental AI integration.
- **Per-Venue Tracking**: API call counting, rate limit monitoring, and latency tracking are implemented for each integrated venue.
- **Real-Time Streaming**: Server-Sent Events (SSE) provide continuous dashboard updates without polling.

**UI/UX Decisions:**
The frontend utilizes a responsive grid layout with a dark theme (Solana Purple/Teal/Orange color scheme). It features 17+ interactive panels including Portfolio Overview, Live Prices, Positions, Funding Rates, PnL Breakdown, Performance Metrics, and Risk Metrics. Chart.js visualizations are used for NAV history, allocation, and PnL breakdowns. Asset-specific icons, gradients, and status indicators enhance user experience.

**Technical Implementations:**
- **`app.py`**: Main Flask REST API server with 60+ endpoints for client initialization, portfolio, pricing, trading, analytics, and experimental AI agent integration.
- **`simulator.py`**: Monte Carlo simulation engine for backtesting and paper trading, including price path generation, PnL tracking, and risk limit enforcement.
- **`infra/`**: Contains low-level blockchain and exchange integrations (Solana RPC, Jupiter, Drift, Hyperliquid, Pyth), with rate limiting, caching (`redis_client.py`), and a unified `perp_venue.py` interface.
- **`data/`**: Handles multi-source price aggregation (`data_fetcher.py`) from 6 venues, analytics (`analytics.py`) for Sharpe/Sortino ratios and drawdowns, and `mock_data.py` for offline development.
- **`strategies/`**: Implements 7 automated yield-generating strategies, including `basis_harvester.py`, `funding_rotator.py`, `perp_spread.py`, `cross_venue_funding_arb.py`, `hedged_basket.py`, `volatility_scaler.py`, and `carry_optimizer.py`.
- **`trading/`**: Manages `paper_account.py` for simulation and `portfolio_config.py` for strategy allocation profiles.
- **`vault/`**: `vault_manager.py` enforces risk limits (max position size, leverage, drawdown, slippage) and handles position sizing.
- **`ai/`**: Experimental `agent_bridge.py` provides 8 tools for automated execution, heavily guarded by `is_simulation` checks and feature flags.
