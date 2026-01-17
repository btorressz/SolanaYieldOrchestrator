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
