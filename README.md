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
