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

---

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
├── app.py                          # Main Flask application 
├── config.py                       # Configuration management 
├── simulator.py                    # Monte Carlo simulation engine 
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
│   ├── paper_account.py            # Simulation paper trading 
│   └── portfolio_config.py         # Portfolio configuration 
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
│   └── index.html                  # Main dashboard 
│
├── static/                         # Frontend Assets
│   ├── main.js                     # Dashboard logic 
│   └── style.css                   # Styling 
│
└── .replit & .gitignore           # Project configuration
```

---

## File Documentation

### Core Application Files

#### `app.py` (Main Flask Server)
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

### Infrastructure Layer (`infra/`)

#### `solana_client.py` (Solana RPC Integration)
Wrapper around Solana RPC for blockchain interactions (balance queries, transaction signing, chain state)

#### `jupiter_client.py` (Jupiter Aggregator)
Handles DEX aggregation: swap quotes, route analysis, MEV risk classification

#### `drift_client.py` (Drift Protocol)
Perpetual trading on Drift: market data, funding rates, margin calculations, position management

#### `hyperliquid_client.py` (Hyperliquid Venue - Optional)
Alternative perpetual trading venue when `HYPERLIQUID_ENABLED=true`

#### `pyth_client.py` (Pyth Oracle)
Decentralized oracle price feeds with deviation calculation and trade gating

#### `perp_venue.py` (Abstract Venue Interface)
Unified interface for perpetual trading across Drift and Hyperliquid

#### `solana_chain_monitor.py` (Chain Monitoring)
Real-time Solana chain state: block monitoring, transaction tracking, network health

#### `redis_client.py` (Caching Layer)
High-performance Redis caching with category-specific TTLs (prices, routes, perp data, metrics)

#### `rate_limiter.py` (Rate Limiting)
Per-venue API rate limit enforcement (RPS and RPM tracking)

#### `priority_router.py` (Multi-Venue Execution)
Intelligent routing of trades across venues with optimization and TWAP execution

#### `metrics_tracker.py` (HFT Metrics)
Tracks high-frequency metrics: latency percentiles, API call counts, execution quality

---

### Data Layer (`data/`)

#### `data_fetcher.py` (Multi-Source Aggregation)
**Core Responsibility:** Aggregate prices from 6 venues with outlier detection and fallback
- Fetches from Jupiter, CoinGecko, Kraken, Hyperliquid, Drift, Pyth
- Calculates weighted average prices
- Detects and removes outliers
- Falls back to mock data if APIs unavailable
- Returns per-venue price breakdown with availability status

#### `analytics.py` (Performance Analytics)
Calculates performance metrics: Sharpe ratio, Sortino ratio, volatility, drawdown, win rate

#### `mock_data.py` (Mock Data Generation)
Generates realistic mock market data for offline development and testing

---

### Strategies Layer (`strategies/`)

#### `basis_harvester.py` (Spot-Perp Basis Trading)
Executes basis harvesting strategy:
- Monitors spot vs perpetual price spreads
- Opens positions when basis exceeds threshold
- Automatically closes when basis normalizes
- Manages position sizing and risk

#### `funding_rotator.py` (Funding Rate Arbitrage)
Rotates positions between markets based on highest funding rates:
- Identifies top funding rate opportunities
- Rotates positions periodically
- Maximizes APY from funding premiums
- Respects risk limits

#### `perp_spread.py` (Cross-Perp Spreads)
Trades spreads between different perpetual contracts (SOL-PERP vs BTC-PERP correlation arbs)

#### `cross_venue_funding_arb.py` (Multi-Venue Arbitrage)
Exploits funding rate differentials between Drift and Hyperliquid

#### `hedged_basket.py` (Hedged Basket)
Maintains correlated asset baskets with dynamic hedging

#### `volatility_scaler.py` (Vol-Scaled Sizing)
Adjusts position sizes based on realized volatility

#### `carry_optimizer.py` (Carry Trade Optimization)
Identifies and optimizes carry trade opportunities across venues

---

### Trading Layer (`trading/`)

#### `paper_account.py` (Paper Trading Engine - 701 lines)
**Purpose:** Simulates trading without real capital
**Key Features:**
- Position tracking and lifecycle
- PnL calculation (unrealized and realized)
- Performance metrics computation
- Risk limit enforcement
- Strategy weight tracking

**Key Methods:**
- `open_position()`: Create new position in simulation
- `close_position()`: Exit existing position
- `get_state()`: Portfolio state snapshot
- `get_metrics()`: Performance metrics
- `get_positions()`: Current holdings

#### `portfolio_config.py` (Portfolio Configuration - 207 lines)
**Purpose:** Manage portfolio strategy weights and settings
- Strategy allocation profiles (Aggressive, Balanced, Conservative)
- Per-strategy weight configuration
- Default settings management

---

### Vault Layer (`vault/`)

#### `vault_manager.py` (Risk Management - 388 lines)
**Purpose:** Enforce risk limits and portfolio constraints
**Key Responsibilities:**
- Risk limit configuration (max position size, max leverage, max drawdown)
- Position sizing calculations
- Portfolio allocation enforcement
- Risk metrics aggregation

#### `mock_hedge_basket.py` (Mock Hedging)
Simulates hedging strategies for portfolio protection

---

### Utility Functions (`utils/`)

#### `logging_utils.py` (Logging System)
Centralized logging configuration with colored output and level management

#### `risk_limits.py` (Risk Definitions)
Risk limit type definitions and validation

---

### Frontend Files

#### `templates/index.html` (Main Dashboard - 899 lines)
**Purpose:** Primary HTML template for dashboard UI
**Key Sections (17+ Panels):**
1. **Portfolio Overview**: NAV, PnL, return %, Sharpe, Sortino, volatility
2. **Live Prices**: Real-time prices with venue badges and oracle status
3. **Portfolio Builder**: Dynamic sliders for asset allocation (12 assets)
4. **Trade Ticket**: Execute trades with market selection
5. **Positions Panel**: Current holdings with entry/exit details
6. **Funding Rates**: Cross-venue funding comparison
7. **Basis Map**: Spot vs perp price spreads
8. **Risk Metrics**: Drawdown, leverage, concentration
9. **Route Quality**: MEV risk assessment for swaps
10. **PnL Breakdown**: Realized vs unrealized gains
11. **Asset Universe**: Asset type filtering and info
12. **Venue Health**: Exchange status and latency
13. **Chain State**: Solana network metrics
14. **HFT Metrics**: Latency and execution quality
15. **Strategy Performance**: Per-strategy breakdown
16. **Activity Log**: Recent trades and actions
17. **Oracle Health**: Pyth deviation status

**Key Features:**
- Responsive grid layout
- Real-time updates via SSE
- Interactive charts with Chart.js
- Asset-specific icons and gradients
- Solana theme (Purple/Teal/Orange)

#### `static/main.js` (Dashboard Logic - 2115 lines)
**Purpose:** Client-side dashboard logic and real-time updates
**Key Functionalities:**
- **Initialization**: Load assets, prices, positions on startup
- **SSE Connection**: Connect to `/events` for real-time updates
- **Rendering**: Update 17+ panels with live data
- **Charts**: Initialize and update Chart.js visualizations
  - NAV history chart
  - Allocation pie chart
  - PnL breakdown chart
  - Target vs actual allocation
- **User Interactions**: 
  - Portfolio allocation sliders
  - Trade execution
  - Scenario stress testing
  - Agent tool execution (when enabled)
- **Data Formatting**: Currency, percentage, number formatting
- **Error Handling**: Graceful fallbacks and retry logic

**Key Functions:**
- `fetchJSON()`: Async API calls
- `initSSE()`: Set up event streaming
- `updatePricesPanel()`: Live price updates
- `updatePortfolioPanel()`: Position updates
- `initCharts()`: Create visualizations
- `executeTrade()`: Submit trade orders
- `updateMetrics()`: Compute performance metrics

#### `static/style.css` (Styling - 2448 lines)
**Purpose:** Complete application styling and theming
**Key Design Elements:**
- **Theme Colors**: 
  - Solana Green: `#14F195`
  - Solana Purple: `#9945FF`
  - Jupiter Teal: `#00D4AA`
  - Drift Orange: `#FF6B35`
- **Layout**:
  - Responsive grid system
  - Sticky header
  - Scrollable panels
  - Card-based component design
- **Animations**:
  - Smooth transitions
  - Loading spinners
  - Status indicators
- **Typography**: Inter font family
- **Dark Mode**: Optimized for dark background with high contrast

**CSS Variables:**
- Background colors (primary, secondary, card, hover)
- Text colors (primary, secondary, muted)
- Gradients (Solana, Jupiter, Drift, mixed)
- Shadows and border radius
- Status indicators (positive, negative, warning)

---

## Architecture

### Layered Design

```
┌─────────────────────────────────────────────┐
│         Frontend (HTML/JS/CSS)              │
│  - Dashboard with 17+ real-time panels      │
│  - Chart.js visualizations                  │
│  - SSE streaming for live updates           │
└──────────────┬──────────────────────────────┘
               │ REST API (60+ endpoints)
┌──────────────▼──────────────────────────────┐
│     Flask Application (app.py)              │
│  - Request routing and validation           │
│  - Portfolio management                     │
│  - Trade execution orchestration            │
└──────────────┬──────────────────────────────┘
┌──────────────▼──────────────────────────────┐
│         Strategy Layer                      │
│  - 7 yield-generating strategies            │
│  - Trade signal generation                  │
└──────────────┬──────────────────────────────┘
┌──────────────▼──────────────────────────────┐
│         Data Layer                          │
│  - Multi-source price aggregation (6 venues)│
│  - Analytics and metrics                    │
│  - Mock data fallback                       │
└──────────────┬──────────────────────────────┘
┌──────────────▼──────────────────────────────┐
│      Infrastructure Layer                   │
│  - Blockchain clients (Solana)              │
│  - Venue clients (Jupiter, Drift, HL, Pyth) │
│  - Rate limiting and routing                │
│  - Redis caching                            │
└──────────────┬──────────────────────────────┘
┌──────────────▼──────────────────────────────┐
│     External Services & Blockchains         │
│  - Solana RPC                               │
│  - Jupiter DEX Aggregator                   │
│  - Drift Protocol                           │
│  - Hyperliquid (optional)                   │
│  - Pyth Oracle                              │
│  - CoinGecko, Kraken APIs                   │
└─────────────────────────────────────────────┘
```

### Data Flow

1. **Price Aggregation**: 6 venues → Data Fetcher → Weighted Average → Frontend
2. **Strategy Execution**: Market Data → Strategies → Signal Generation → Risk Manager → Executor → Venue
3. **Real-Time Updates**: Backend events → SSE → Frontend → Dashboard refresh
4. **Paper Trading**: User action → Paper Account → Position tracking → Metrics calculation

---

## Supported Assets

| Asset | Type | Decimals | Venues | Mint Address |
|-------|------|----------|--------|--------------|
| SOL | native | 9 | Drift + HL | So111... |
| BTC | wrapped | 8 | Drift + HL | 9n4nb... |
| ETH | wrapped | 8 | Drift + HL | 7vfCX... |
| mSOL | LST | 9 | Spot only | mSoLz... |
| BONK | meme | 5 | Jupiter only | DezXA... |
| XRP | wrapped | 6 | Jupiter only | 7RMr7... |
| USDC | stablecoin | 6 | All | EPjFW... |
| USDT | stablecoin | 6 | All | Es9vM... |
| JTO | governance | 9 | Jupiter only | jtojtom... |
| JUP | governance | 6 | Jupiter only | JUPyw... |
| ORCA | dex | 6 | Jupiter only | orcaEK... |
| RAY | dex | 6 | Jupiter only | 4k3Dy... |

---

## Multi-Venue Integration

### Perpetual Trading Venues

**Drift Protocol**
- Markets: SOL-PERP, BTC-PERP, ETH-PERP
- Funding rates and mark prices
- Margin calculations
- Always enabled

**Hyperliquid** (Optional)
- Markets: SOL, BTC, ETH
- Enable with: `HYPERLIQUID_ENABLED=true`
- Higher leverage available
- Cross-venue arbitrage opportunities

### Price Sources

1. **Jupiter Aggregator**: Most comprehensive DEX aggregation
2. **Pyth Network**: Decentralized oracle (blocks trades if deviation >100 bps)
3. **CoinGecko**: Reference pricing and comparison
4. **Kraken**: CEX price validation
5. **Drift Protocol**: Perpetual mark prices
6. **Hyperliquid**: Alternative perp venue prices

---

## API Endpoints

### Portfolio Management
- `GET /api/portfolio/state` - Portfolio state snapshot
- `GET /api/portfolio/config` - Current configuration
- `POST /api/portfolio/config` - Update portfolio config
- `GET /api/user/positions` - Current positions

### Pricing
- `GET /api/prices` - Aggregated prices from all venues
- `GET /api/venue/basis-map` - Spot vs perp basis
- `GET /api/venue/status` - Venue health status
- `GET /api/cross-venue/funding` - Funding rates across venues

### Trading
- `POST /api/trade/execute` - Execute trade (paper or live)
- `POST /api/trade/close` - Close position
- `POST /api/trade/liquidate` - Force liquidate

### Analytics
- `GET /api/metrics` - Portfolio performance metrics
- `GET /api/hft/latency` - Execution latency
- `GET /api/jupiter/route-quality` - MEV risk analysis
- `GET /api/oracle/pyth-health` - Oracle deviation status

### Streaming
- `GET /events` - Server-Sent Events for real-time updates

### AI Agent (Experimental)
- `GET /api/agent/tools` - List available tools
- `POST /api/agent/execute` - Execute tool
- `POST /api/agent/run-sample` - Run demo flow

---

## Configuration

### Environment Variables

```bash
# Operating Mode
MODE=simulation                    # simulation | live
MARKET_DATA_MODE=auto             # auto | live | mock

# Blockchain
SOLANA_RPC_URL=https://...        # Solana mainnet RPC
SOLANA_WS_URL=wss://...           # WebSocket URL
SOLANA_PRIVATE_KEY=               # Optional: for live trading

# Venue APIs
HYPERLIQUID_ENABLED=false         # Enable cross-venue trading
PYTH_ENABLED=true                 # Enable oracle monitoring
MAX_ORACLE_DEVIATION_BPS=100      # Trade blocking threshold

# Agent Kit (Phase 2, Experimental)
AGENT_KIT_ENABLED=false           # Disable by default
AGENT_KIT_LIVE_TRADING=false      # Future live trading flag

# Strategy Parameters
BASIS_ENTRY_THRESHOLD_BPS=50      # Basis trading threshold
FUNDING_MIN_APY_THRESHOLD=5.0     # Min funding to trade
ROTATION_EPOCH_SECONDS=3600       # Funding rotation interval

# Risk Limits
MAX_POSITION_SIZE_USD=10000       # Per-position limit
MAX_LEVERAGE=5.0                  # Margin multiplier
MAX_DRAWDOWN_PCT=10.0             # Drawdown limit
MAX_SLIPPAGE_BPS=100              # Slippage tolerance

# Rate Limiting
RATE_LIMIT_JUPITER_RPS=10         # Jupiter requests/sec
RATE_LIMIT_DRIFT_RPS=10           # Drift requests/sec
RATE_LIMIT_HYPERLIQUID_RPS=10     # Hyperliquid requests/sec
```

---

## Installation & Usage

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set environment variables (or create .env file)
export MODE=simulation
export MARKET_DATA_MODE=auto

# Run application
python app.py
```

### Access Dashboard
```
http://localhost:5000
```

### Key Features to Try
1. **Live Prices Panel**: View real-time prices from 6 venues
2. **Portfolio Builder**: Drag sliders to adjust asset allocation
3. **Trade Ticket**: Execute simulated trades
4. **Performance Charts**: Monitor strategy performance
5. **Risk Monitoring**: Track drawdown and leverage
6. **Agent Tools** (when `AGENT_KIT_ENABLED=true`): Execute automated strategies

---

## Technology Stack

### Backend
- **Framework**: Flask 2.0+ (Python web framework)
- **Blockchain**: Solana RPC (transaction building, state monitoring)
- **Exchanges**: Jupiter, Drift, Hyperliquid, Pyth
- **Data**: Redis (caching), CoinGecko/Kraken (reference prices)
- **Async**: Python asyncio for concurrent operations
- **Logging**: Custom logging with color formatting

### Frontend
- **HTML5**: Semantic markup with responsive design
- **JavaScript**: Vanilla JS with async/await
- **Charts**: Chart.js for real-time visualizations
- **Styling**: CSS3 with CSS variables and gradients
- **Streaming**: Server-Sent Events (SSE)

### Data & Analytics
- **Price Aggregation**: 6-venue weighted averaging with outlier detection
- **Performance Metrics**: Sharpe ratio, Sortino ratio, volatility, max drawdown
- **Simulation**: Monte Carlo engine for backtesting

### DevOps
- **Python 3.11+**
- **Requirements**: Flask, requests, websockets, redis, solders (Solana SDK)
- **Port**: 5000 (default)
- **CORS**: Enabled for frontend integration

---

## Safety & Risk Management

- **Paper Trading Default**: All trades simulated unless explicitly switched to live mode
- **Oracle Gating**: Trades blocked if Pyth deviation exceeds 100 bps
- **Position Limits**: Max position size and leverage enforced
- **Risk Limits**: Drawdown monitoring and enforcement
- **Agent Kit Gating**: AI agents require simulation mode + explicit flag
- **Rate Limiting**: Per-venue API rate limits to prevent abuse

---

## Disclaimer

This platform is provided for educational and research purposes. Always perform your own due diligence when trading on real capital. Past performance does not guarantee future results.



---
