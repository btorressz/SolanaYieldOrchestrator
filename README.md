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

---
