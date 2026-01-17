#!/usr/bin/env python3
"""
Solana Yield Orchestrator - Flask API Server

Provides REST API endpoints for:
- Portfolio metrics and status
- Trade execution (simulated and live)
- Strategy configuration and profiles
- Real-time updates via SSE
- Scenario stress-testing
"""
import os
import time
import threading
import json
from datetime import datetime
from queue import Queue

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

from config import Config
from infra.solana_client import SolanaClient
from infra.jupiter_client import JupiterClient, JupiterPriceAPI, EnhancedJupiterClient, COMMON_TOKENS
from infra.drift_client import DriftClientWrapper
from infra.priority_router import PriorityRouter, TWAPExecutor
from infra.redis_client import is_redis_available
from data.data_fetcher import DataFetcher
from data.analytics import Analytics
from vault.vault_manager import VaultManager
from simulator import Simulator
from trading.paper_account import PaperAccount, get_paper_account, reset_paper_account
from trading.portfolio_config import (
    PortfolioConfig, get_portfolio_config, set_portfolio_config, 
    update_portfolio_config, get_default_weights_from_profile
)
from utils.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
CORS(app)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

solana_client = None
jupiter_client = None
enhanced_jupiter = None
drift_client = None
priority_router = None
twap_executor = None
data_fetcher = None
vault_manager = None
analytics = Analytics()

orchestrator_running = False
last_update_time = 0
current_snapshot = None
metrics_history = []

sse_clients = []

sim_config = {
    "initial_nav": 10000.0,
    "profile": None,
    "strategies": {
        "basis": True,
        "funding": True,
        "carry": False,
        "volatility": False,
        "basket": False
    },
    "thresholds": {
        "basis_entry_bps": Config.BASIS_ENTRY_THRESHOLD_BPS,
        "basis_exit_bps": Config.BASIS_EXIT_THRESHOLD_BPS,
        "funding_min_apy": Config.FUNDING_MIN_APY_THRESHOLD,
        "funding_top_n": Config.FUNDING_TOP_N_MARKETS
    },
    "risk_limits": {
        "max_position_size_usd": Config.MAX_POSITION_SIZE_USD,
        "max_leverage": 5.0,
        "max_drawdown_pct": Config.MAX_DRAWDOWN_PCT,
        "max_slippage_bps": Config.MAX_SLIPPAGE_BPS
    },
    "allocations": Config.get_allocations()
}

def initialize_clients():
    global solana_client, jupiter_client, enhanced_jupiter, drift_client, priority_router, twap_executor, data_fetcher, vault_manager
    
    logger.info(f"Initializing clients in {Config.MODE} mode...")
    
    solana_client = SolanaClient()
    jupiter_client = JupiterClient()
    enhanced_jupiter = EnhancedJupiterClient()
    drift_client = DriftClientWrapper(solana_client)
    
    priority_router = PriorityRouter(
        solana_client=solana_client,
        jupiter_client=jupiter_client,
        drift_client=drift_client
    )
    
    twap_executor = TWAPExecutor(priority_router, jupiter_client)
    
    from infra.hyperliquid_client import hyperliquid_client as hl_client
    
    data_fetcher = DataFetcher(
        solana_client=solana_client,
        jupiter_client=jupiter_client,
        drift_client=drift_client,
        hyperliquid_client=hl_client
    )
    
    vault_manager = VaultManager(initial_capital=sim_config["initial_nav"])
    
    get_paper_account(sim_config["initial_nav"])
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(drift_client.initialize())
    except Exception as e:
        logger.warning(f"Drift client initialization warning: {e}")
    finally:
        loop.close()
    
    logger.info("All clients initialized successfully")

def broadcast_sse_event(event_type: str, data: dict):
    """Broadcast an SSE event to all connected clients."""
    global sse_clients
    message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead_clients = []
    for client_queue in sse_clients:
        try:
            client_queue.put_nowait(message)
        except:
            dead_clients.append(client_queue)
    for client in dead_clients:
        if client in sse_clients:
            sse_clients.remove(client)

def orchestrator_loop():
    global orchestrator_running, last_update_time, current_snapshot, metrics_history
    
    logger.info("Starting orchestrator loop...")
    orchestrator_running = True
    
    while orchestrator_running:
        try:
            current_snapshot = data_fetcher.get_market_snapshot()
            
            actions = vault_manager.update(current_snapshot)
            
            if actions:
                results = vault_manager.execute_pending_actions(priority_router)
                for result in results:
                    logger.info(f"Action result: {result}")
            
            status = vault_manager.get_status()
            
            metrics_entry = {
                "timestamp": time.time(),
                "nav": status["vault_state"]["total_nav"],
                "pnl": status["portfolio_metrics"]["total_pnl"],
                "pnl_pct": status["portfolio_metrics"]["total_pnl_pct"]
            }
            metrics_history.append(metrics_entry)
            
            if len(metrics_history) > 1000:
                metrics_history = metrics_history[-500:]
            
            paper_account = get_paper_account()
            prices = {}
            if data_fetcher:
                for symbol in ["SOL", "BTC", "ETH"]:
                    agg = data_fetcher.get_aggregated_price(symbol)
                    if agg:
                        prices[symbol] = agg.get("price", 100.0)
            paper_account.update_prices(prices)
            
            analytics.record_nav(paper_account.current_nav)
            paper_account.record_exposure()
            
            broadcast_sse_event("metrics", {
                "nav": paper_account.current_nav,
                "pnl": paper_account.total_pnl,
                "pnl_pct": paper_account.total_return_pct,
                "timestamp": time.time()
            })
            
            broadcast_sse_event("prices", prices)
            
            last_update_time = time.time()
            
        except Exception as e:
            logger.error(f"Orchestrator loop error: {e}")
        
        time.sleep(10)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def index():
    return render_template("index.html", mode=Config.MODE)

@app.route("/api/status")
def api_status():
    global last_update_time, orchestrator_running, current_snapshot
    
    data_mode_info = {"data_mode": "unknown"}
    source_status = {}
    
    if data_fetcher:
        source_status = data_fetcher.get_source_status()
        if current_snapshot and current_snapshot.metadata:
            data_mode_info = {
                "data_mode": current_snapshot.metadata.get("data_mode", "unknown"),
                "mock_sources": current_snapshot.metadata.get("mock_sources", []),
                "live_sources": current_snapshot.metadata.get("sources", []),
            }
            if "mock_reason" in current_snapshot.metadata:
                data_mode_info["mock_reason"] = current_snapshot.metadata["mock_reason"]
    
    return jsonify({
        "status": "running" if orchestrator_running else "stopped",
        "mode": Config.MODE,
        "market_data_mode": Config.MARKET_DATA_MODE,
        "data_mode_info": data_mode_info,
        "source_status": source_status,
        "redis_available": is_redis_available(),
        "last_update": last_update_time,
        "last_update_ago": time.time() - last_update_time if last_update_time > 0 else None,
        "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else 0
    })

@app.route("/api/prices")
def api_prices():
    """
    Returns aggregated prices for all supported assets with per-venue breakdown.
    
    Response includes:
    - Per-venue prices (Jupiter, CoinGecko, Kraken, Hyperliquid, Drift)
    - Pyth oracle data (pyth_price, oracle_deviation_bps, oracle_status)
    - venues object with availability indicators for each venue
    """
    try:
        jupiter_prices = {}
        coingecko_prices = {}
        kraken_prices = {}
        hyperliquid_prices = {}
        pyth_prices = {}
        drift_prices = {}
        
        if data_fetcher:
            jup = data_fetcher.get_jupiter_prices()
            jupiter_prices = {k: {"price": v.price, "source": v.source} for k, v in jup.items()}
            
            cg = data_fetcher.get_coingecko_prices()
            coingecko_prices = {k: {"price": v.price, "change_24h": v.change_24h} for k, v in cg.items()}
            
            kr = data_fetcher.get_kraken_prices()
            kraken_prices = {k: {"price": v.price, "change_24h": v.change_24h} for k, v in kr.items()}
            
            hl = data_fetcher.get_hyperliquid_prices()
            hyperliquid_prices = {k: {"price": v.price, "source": v.source, "volume_24h": v.volume_24h} for k, v in hl.items()}
            
            perp_prices, _ = data_fetcher.get_drift_data()
            for market, mark_price in perp_prices.items():
                base_symbol = market.replace("-PERP", "")
                if base_symbol in Config.SUPPORTED_ASSETS:
                    drift_prices[base_symbol] = {"mark_price": mark_price, "market": market}
        
        if Config.PYTH_ENABLED:
            try:
                from infra.pyth_client import get_pyth_client
                pyth = get_pyth_client()
                all_pyth = pyth.get_all_prices()
                for symbol, pyth_data in all_pyth.items():
                    if pyth_data:
                        pyth_prices[symbol] = {
                            "price": pyth_data.price,
                            "confidence": pyth_data.confidence,
                            "timestamp": pyth_data.timestamp
                        }
            except Exception as e:
                logger.debug(f"Pyth prices fetch failed: {e}")
        
        aggregated = {}
        if data_fetcher:
            for symbol in Config.SUPPORTED_ASSETS:
                agg = data_fetcher.get_aggregated_price(symbol, include_pyth=False)
                if agg:
                    composite_price = agg.get("average_price", 0)
                    
                    pyth_price = None
                    oracle_deviation_bps = None
                    oracle_status = "unknown"
                    
                    if Config.PYTH_ENABLED and symbol in pyth_prices:
                        pyth_price = pyth_prices[symbol].get("price")
                        if pyth_price and composite_price and composite_price > 0:
                            deviation_pct = abs(pyth_price - composite_price) / composite_price
                            oracle_deviation_bps = round(deviation_pct * 10000, 2)
                            
                            if oracle_deviation_bps <= 25:
                                oracle_status = "clean"
                            elif oracle_deviation_bps <= 100:
                                oracle_status = "watch"
                            else:
                                oracle_status = "flagged"
                    
                    jup_available = symbol in jupiter_prices
                    cg_available = symbol in coingecko_prices
                    kr_available = symbol in kraken_prices
                    hl_available = symbol in hyperliquid_prices
                    drift_available = symbol in drift_prices
                    pyth_available = symbol in pyth_prices
                    
                    venues = {
                        "jupiter": {
                            "price": jupiter_prices[symbol]["price"] if jup_available else None,
                            "available": jup_available
                        },
                        "coingecko": {
                            "price": coingecko_prices[symbol]["price"] if cg_available else None,
                            "available": cg_available
                        },
                        "kraken": {
                            "price": kraken_prices[symbol]["price"] if kr_available else None,
                            "available": kr_available
                        },
                        "hyperliquid": {
                            "price": hyperliquid_prices[symbol]["price"] if hl_available else None,
                            "available": hl_available
                        },
                        "drift": {
                            "mark_price": drift_prices[symbol]["mark_price"] if drift_available else None,
                            "available": drift_available
                        },
                        "pyth": {
                            "price": pyth_price,
                            "available": pyth_available
                        }
                    }
                    
                    sources = {}
                    if jup_available:
                        sources["jupiter"] = jupiter_prices[symbol]["price"]
                    if cg_available:
                        sources["coingecko"] = coingecko_prices[symbol]["price"]
                    if kr_available:
                        sources["kraken"] = kraken_prices[symbol]["price"]
                    if hl_available:
                        sources["hyperliquid"] = hyperliquid_prices[symbol]["price"]
                    if pyth_price:
                        sources["pyth"] = pyth_price
                    
                    agg["price"] = composite_price
                    agg["pyth_price"] = pyth_price
                    agg["oracle_deviation_bps"] = oracle_deviation_bps
                    agg["oracle_status"] = oracle_status
                    agg["venues"] = venues
                    agg["sources"] = sources
                    agg["has_hyperliquid"] = symbol in Config.ASSET_HL_SYMBOLS
                    agg["has_drift"] = symbol in Config.ASSET_DRIFT_SYMBOLS
                    agg["cross_venue_tradable"] = (symbol in Config.ASSET_HL_SYMBOLS) and (symbol in Config.ASSET_DRIFT_SYMBOLS)
                    
                    aggregated[symbol] = agg
        
        return jsonify({
            "timestamp": time.time(),
            "supported_assets": Config.SUPPORTED_ASSETS,
            "pyth_enabled": Config.PYTH_ENABLED,
            "hyperliquid_enabled": Config.HYPERLIQUID_ENABLED,
            "jupiter": jupiter_prices,
            "coingecko": coingecko_prices,
            "kraken": kraken_prices,
            "hyperliquid": hyperliquid_prices,
            "drift": drift_prices,
            "pyth": pyth_prices,
            "aggregated": aggregated
        })
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/assets", methods=["GET", "POST"])
def api_assets():
    try:
        session_id = request.cookies.get("session_id", "default")
        config = get_portfolio_config(session_id)
        
        if request.method == "POST":
            data = request.get_json() or {}
            symbols = data.get("symbols", [])
            
            valid_symbols = [s for s in symbols if s in Config.SUPPORTED_ASSETS]
            if not valid_symbols:
                valid_symbols = ["SOL", "BTC", "ETH"]
            
            config, error = update_portfolio_config(
                session_id, 
                enabled_assets=valid_symbols
            )
            
            if error:
                return jsonify({"success": False, "error": error}), 400
            
            supported_assets_info = [Config.get_asset_info(s) for s in Config.SUPPORTED_ASSETS]
            enabled_assets_info = [Config.get_asset_info(s) for s in config.enabled_assets]
            
            return jsonify({
                "success": True,
                "enabled_assets": config.enabled_assets,
                "enabled_assets_info": enabled_assets_info,
                "supported_assets": Config.SUPPORTED_ASSETS,
                "supported_assets_info": supported_assets_info,
                "message": f"Updated asset universe to {len(valid_symbols)} assets"
            })
        
        supported_assets_info = [Config.get_asset_info(s) for s in Config.SUPPORTED_ASSETS]
        enabled_assets_info = [Config.get_asset_info(s) for s in config.enabled_assets]
        
        return jsonify({
            "success": True,
            "supported_assets": Config.SUPPORTED_ASSETS,
            "supported_assets_info": supported_assets_info,
            "enabled_assets": config.enabled_assets,
            "enabled_assets_info": enabled_assets_info,
            "asset_weights": config.asset_weights,
            "hyperliquid_enabled": Config.HYPERLIQUID_ENABLED,
            "pyth_enabled": Config.PYTH_ENABLED
        })
        
    except Exception as e:
        logger.error(f"Assets API error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/metrics")
def api_metrics():
    try:
        if not vault_manager:
            return jsonify({"error": "Vault manager not initialized"}), 500
        
        status = vault_manager.get_status()
        allocations = vault_manager.get_allocations()
        
        funding_rates = {}
        if current_snapshot:
            for market, data in current_snapshot.funding_rates.items():
                if hasattr(data, 'funding_rate'):
                    funding_rates[market] = {
                        "rate": data.funding_rate,
                        "apy": data.funding_apy
                    }
                else:
                    funding_rates[market] = data
        
        paper_account = get_paper_account()
        extended_metrics = analytics.calculate_extended_metrics()
        pnl_breakdown = paper_account.get_pnl_breakdown()
        max_exposure = paper_account.get_max_exposure_by_market()
        
        data_mode_info = {}
        if current_snapshot and current_snapshot.metadata:
            data_mode_info = {
                "data_mode": current_snapshot.metadata.get("data_mode", "unknown"),
                "mock_sources": current_snapshot.metadata.get("mock_sources", []),
                "live_sources": current_snapshot.metadata.get("sources", []),
            }
        
        return jsonify({
            "timestamp": time.time(),
            "vault": status["vault_state"],
            "portfolio": status["portfolio_metrics"],
            "strategies": status["strategies"],
            "allocations": allocations,
            "risk": status["risk"],
            "funding_rates": funding_rates,
            "cycle_count": status["cycle_count"],
            "mode": Config.MODE,
            "market_data_mode": Config.MARKET_DATA_MODE,
            "data_mode_info": data_mode_info,
            "extended_metrics": {
                "sharpe_ratio": extended_metrics.get("sharpe_ratio", 0),
                "sortino_ratio": extended_metrics.get("sortino_ratio", 0),
                "nav_volatility": extended_metrics.get("nav_volatility", 0),
                "max_drawdown": extended_metrics.get("max_drawdown", 0)
            },
            "pnl_breakdown": pnl_breakdown,
            "max_exposure_by_market": max_exposure
        })
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/simulate", methods=["GET", "POST"])
def api_simulate():
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            steps = data.get("steps", 50)
            capital = data.get("capital", 10000.0)
        else:
            steps = int(request.args.get("steps", 50))
            capital = float(request.args.get("capital", 10000.0))
        
        steps = min(steps, 500)
        
        simulator = Simulator(initial_capital=capital)
        results = simulator.run_simulation(num_steps=steps)
        
        return jsonify({
            "success": True,
            "results": results
        })
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/config")
def api_config():
    return jsonify({
        "mode": Config.MODE,
        "allocations": Config.get_allocations(),
        "thresholds": {
            "basis_entry_bps": Config.BASIS_ENTRY_THRESHOLD_BPS,
            "basis_exit_bps": Config.BASIS_EXIT_THRESHOLD_BPS,
            "funding_min_apy": Config.FUNDING_MIN_APY_THRESHOLD,
            "funding_top_n": Config.FUNDING_TOP_N_MARKETS,
            "rotation_epoch_seconds": Config.ROTATION_EPOCH_SECONDS
        },
        "risk_limits": {
            "max_position_size_usd": Config.MAX_POSITION_SIZE_USD,
            "max_slippage_bps": Config.MAX_SLIPPAGE_BPS,
            "max_drawdown_pct": Config.MAX_DRAWDOWN_PCT
        },
        "priority_fees": {
            "cheap": Config.PRIORITY_FEE_CHEAP,
            "balanced": Config.PRIORITY_FEE_BALANCED,
            "fast": Config.PRIORITY_FEE_FAST
        },
        "available_profiles": Config.get_all_profiles()
    })

@app.route("/api/history")
def api_history():
    limit = int(request.args.get("limit", 100))
    
    return jsonify({
        "timestamp": time.time(),
        "metrics_history": metrics_history[-limit:],
        "count": len(metrics_history[-limit:])
    })

@app.route("/api/snapshot")
def api_snapshot():
    if current_snapshot:
        return jsonify({
            "timestamp": time.time(),
            "snapshot": current_snapshot.to_dict()
        })
    else:
        return jsonify({
            "timestamp": time.time(),
            "snapshot": None,
            "message": "No snapshot available yet"
        })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/api/trade/simulate", methods=["POST"])
def api_trade_simulate():
    try:
        data = request.get_json() or {}
        
        venue = data.get("venue", "jupiter").lower()
        market = data.get("market", "SOL/USDC")
        side = data.get("side", "buy").lower()
        size = float(data.get("size", 0))
        slippage_bps = int(data.get("slippage_bps", 50))
        priority_profile = data.get("priority_profile", "balanced")
        
        if size <= 0:
            return jsonify({"success": False, "error": "Size must be positive"}), 400
        
        paper_account = get_paper_account()
        
        prices = {}
        if data_fetcher:
            for symbol in ["SOL", "BTC", "ETH"]:
                agg = data_fetcher.get_aggregated_price(symbol)
                if agg:
                    prices[symbol] = agg.get("price", 100.0)
        
        if venue == "jupiter":
            parts = market.split("/")
            base = parts[0].upper() if parts else "SOL"
            price = prices.get(base, 100.0)
            
            slippage_cost = price * (slippage_bps / 10000)
            executed_price = price + slippage_cost if side in ["buy", "long"] else price - slippage_cost
            
            fees = size * executed_price * 0.0005
            
            result = paper_account.execute_spot_trade(
                market=market,
                side=side,
                size=size,
                price=executed_price,
                fees=fees,
                slippage_bps=slippage_bps,
                simulated=True
            )
        else:
            base = market.replace("-PERP", "").upper()
            price = prices.get(base, 100.0)
            
            slippage_cost = price * (slippage_bps / 10000)
            executed_price = price + slippage_cost if side in ["long"] else price - slippage_cost
            
            leverage = float(data.get("leverage", 1.0))
            fees = size * executed_price * 0.001
            
            result = paper_account.execute_perp_trade(
                market=market,
                side=side,
                size=size,
                price=executed_price,
                leverage=leverage,
                fees=fees,
                slippage_bps=slippage_bps,
                simulated=True
            )
        
        if result["success"]:
            result["filled_price"] = executed_price
            result["account_status"] = paper_account.get_status()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Trade simulation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/trade/live", methods=["POST"])
def api_trade_live():
    if Config.MODE != "live":
        return jsonify({
            "success": False,
            "error": "Live trading is disabled. Set MODE=live to enable."
        }), 403
    
    try:
        data = request.get_json() or {}
        
        return jsonify({
            "success": False,
            "error": "Live trading implementation pending. Use simulation mode for now.",
            "mode": Config.MODE
        }), 501
        
    except Exception as e:
        logger.error(f"Live trade error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/user/positions")
def api_user_positions():
    try:
        paper_account = get_paper_account()
        
        prices = {}
        if data_fetcher:
            for symbol in ["SOL", "BTC", "ETH"]:
                agg = data_fetcher.get_aggregated_price(symbol)
                if agg:
                    prices[symbol] = agg.get("price", 100.0)
        
        paper_account.update_prices(prices)
        
        return jsonify({
            "success": True,
            "account": paper_account.get_status(),
            "positions": paper_account.get_positions_summary(),
            "risk_metrics": paper_account.get_risk_metrics(),
            "trade_history": paper_account.get_trade_history(limit=20)
        })
        
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sim/config", methods=["GET", "POST"])
def api_sim_config():
    global sim_config, vault_manager
    
    if request.method == "GET":
        return jsonify({
            "success": True,
            "config": sim_config,
            "available_profiles": Config.get_all_profiles()
        })
    
    try:
        data = request.get_json() or {}
        
        if "profile" in data:
            profile_name = data["profile"].lower()
            profile = Config.get_strategy_profile(profile_name)
            
            sim_config["profile"] = profile_name
            sim_config["strategies"] = profile["strategies"].copy()
            sim_config["thresholds"] = profile["thresholds"].copy()
            sim_config["risk_limits"] = profile["risk_limits"].copy()
            sim_config["allocations"] = profile["allocations"].copy()
            
            logger.info(f"Applied strategy profile: {profile_name}")
        
        if "initial_nav" in data:
            sim_config["initial_nav"] = float(data["initial_nav"])
            reset_paper_account(sim_config["initial_nav"])
            if vault_manager:
                vault_manager = VaultManager(initial_capital=sim_config["initial_nav"])
        
        if "strategies" in data:
            for strategy, enabled in data["strategies"].items():
                if strategy in sim_config["strategies"]:
                    sim_config["strategies"][strategy] = bool(enabled)
        
        if "thresholds" in data:
            for key, value in data["thresholds"].items():
                if key in sim_config["thresholds"]:
                    sim_config["thresholds"][key] = float(value)
        
        if "risk_limits" in data:
            for key, value in data["risk_limits"].items():
                if key in sim_config["risk_limits"]:
                    sim_config["risk_limits"][key] = float(value)
        
        return jsonify({
            "success": True,
            "config": sim_config,
            "message": "Configuration updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Config update error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sim/reset", methods=["POST"])
def api_sim_reset():
    try:
        data = request.get_json() or {}
        initial_nav = float(data.get("initial_nav", sim_config["initial_nav"]))
        
        paper_account = reset_paper_account(initial_nav)
        analytics.reset()
        
        return jsonify({
            "success": True,
            "message": f"Paper account reset with NAV=${initial_nav:,.2f}",
            "account": paper_account.get_status()
        })
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    """
    Run a stress-test scenario on the current portfolio.
    
    Request body:
    {
        "price_shocks": {"SOL": -0.2, "BTC": -0.1},  // -20% SOL, -10% BTC
        "funding_shocks": {"SOL-PERP": 0.5}  // Optional: +50% funding
    }
    
    Returns:
    {
        "shocked_nav": ...,
        "nav_change_pct": ...,
        "positions": [...],
        "liq_warnings": [...]
    }
    """
    try:
        data = request.get_json() or {}
        
        price_shocks = data.get("price_shocks", {})
        funding_shocks = data.get("funding_shocks", {})
        
        paper_account = get_paper_account()
        
        prices = {}
        if data_fetcher:
            for symbol in ["SOL", "BTC", "ETH"]:
                agg = data_fetcher.get_aggregated_price(symbol)
                if agg:
                    prices[symbol] = agg.get("price", 100.0)
        paper_account.update_prices(prices)
        
        all_positions = []
        for pos in paper_account.spot_positions.values():
            all_positions.append(pos.to_dict())
        for pos in paper_account.perp_positions.values():
            all_positions.append(pos.to_dict())
        
        scenario_result = analytics.run_scenario(
            positions=all_positions,
            price_shocks=price_shocks,
            funding_shocks=funding_shocks,
            current_nav=paper_account.current_nav
        )
        
        return jsonify({
            "success": True,
            "scenario": scenario_result
        })
        
    except Exception as e:
        logger.error(f"Scenario error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pnl/breakdown")
def api_pnl_breakdown():
    """Get detailed PnL breakdown by component."""
    try:
        paper_account = get_paper_account()
        
        return jsonify({
            "success": True,
            "breakdown": paper_account.get_pnl_breakdown(),
            "history": paper_account.get_pnl_component_history(hours=24),
            "max_exposure": paper_account.get_max_exposure_by_market()
        })
        
    except Exception as e:
        logger.error(f"PnL breakdown error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/events")
def events():
    """
    Server-Sent Events endpoint for real-time updates.
    
    Streams the following event types:
    - prices: Latest price data
    - metrics: NAV, PnL updates
    - positions: Position changes
    """
    def event_stream():
        client_queue = Queue()
        sse_clients.append(client_queue)
        
        try:
            yield f"event: connected\ndata: {json.dumps({'status': 'connected', 'timestamp': time.time()})}\n\n"
            
            while True:
                try:
                    message = client_queue.get(timeout=30)
                    yield message
                except:
                    yield f"event: heartbeat\ndata: {json.dumps({'timestamp': time.time()})}\n\n"
        except GeneratorExit:
            pass
        except Exception as e:
            logger.debug(f"SSE connection closed: {e}")
        finally:
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)
    
    response = Response(event_stream(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Connection'] = 'keep-alive'
    return response


@app.route("/api/jupiter/route-info")
def api_jupiter_route_info():
    try:
        market = request.args.get("market", "SOL/USDC")
        size = float(request.args.get("size", 1.0))
        
        parts = market.split("/")
        input_symbol = parts[0].upper() if parts else "SOL"
        output_symbol = parts[1].upper() if len(parts) > 1 else "USDC"
        
        input_mint = COMMON_TOKENS.get(input_symbol, COMMON_TOKENS["SOL"])
        output_mint = COMMON_TOKENS.get(output_symbol, COMMON_TOKENS["USDC"])
        
        decimals = 9 if input_symbol in ["SOL", "mSOL"] else 6
        amount = int(size * (10 ** decimals))
        
        if enhanced_jupiter:
            route_info = enhanced_jupiter.get_route_info(input_mint, output_mint, amount)
            
            if route_info:
                return jsonify({
                    "success": True,
                    "route": enhanced_jupiter.to_route_info_dict(route_info)
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to get route info",
                    "fallback": {
                        "market": market,
                        "estimated_price": 100.0,
                        "route_labels": ["simulated"],
                        "is_direct": True
                    }
                })
        else:
            return jsonify({
                "success": False,
                "error": "Jupiter client not initialized"
            }), 500
            
    except Exception as e:
        logger.error(f"Route info error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/sanity-check")
def api_jupiter_sanity_check():
    try:
        if not enhanced_jupiter or not data_fetcher:
            return jsonify({"success": False, "error": "Clients not initialized"}), 500
        
        prices = {
            "jupiter": {},
            "coingecko": {},
            "kraken": {}
        }
        
        for symbol in ["SOL", "BTC", "ETH"]:
            jup = data_fetcher.get_jupiter_prices()
            if symbol in jup:
                prices["jupiter"][symbol] = jup[symbol].price
            
            cg = data_fetcher.get_coingecko_prices()
            if symbol in cg:
                prices["coingecko"][symbol] = cg[symbol].price
            
            kr = data_fetcher.get_kraken_prices()
            if symbol in kr:
                prices["kraken"][symbol] = kr[symbol].price
        
        sanity_result = enhanced_jupiter.compare_cross_venue_prices(prices)
        
        return jsonify({
            "success": True,
            "sanity_check": sanity_result
        })
        
    except Exception as e:
        logger.error(f"Sanity check error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/perp/risk")
def api_perp_risk():
    try:
        paper_account = get_paper_account()
        
        perp_positions = [pos.to_dict() for pos in paper_account.perp_positions.values()]
        
        risk_metrics = analytics.calculate_perp_risk_metrics(perp_positions)
        
        liquidation_ladders = {}
        for pos in perp_positions:
            market = pos.get("market", "unknown")
            ladder = analytics.calculate_partial_liquidation_ladder(pos)
            liquidation_ladders[market] = ladder
        
        return jsonify({
            "success": True,
            "risk_metrics": risk_metrics,
            "liquidation_ladders": liquidation_ladders,
            "paper_account_risk": paper_account.get_risk_metrics()
        })
        
    except Exception as e:
        logger.error(f"Perp risk error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/portfolio/config", methods=["GET", "POST"])
def api_portfolio_config():
    """
    GET: Returns the current PortfolioConfig for this session.
    POST: Updates the PortfolioConfig with provided values.
    """
    session_id = "default"
    
    if request.method == "GET":
        try:
            config = get_portfolio_config(session_id)
            return jsonify({
                "success": True,
                "initial_nav": config.initial_nav,
                "strategy_weights": config.strategy_weights,
                "asset_weights": config.asset_weights,
                "session_id": session_id
            })
        except Exception as e:
            logger.error(f"Portfolio config GET error: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    try:
        data = request.get_json() or {}
        
        initial_nav = data.get("initial_nav")
        strategy_weights = data.get("strategy_weights")
        asset_weights = data.get("asset_weights")
        
        if initial_nav is not None:
            initial_nav = float(initial_nav)
            if initial_nav <= 0:
                return jsonify({"success": False, "error": "initial_nav must be greater than 0"}), 400
        
        if strategy_weights:
            if not isinstance(strategy_weights, dict):
                return jsonify({"success": False, "error": "strategy_weights must be an object"}), 400
            for key in ["basis", "funding", "cash"]:
                if key not in strategy_weights:
                    strategy_weights[key] = 0.0
            total = sum(strategy_weights.values())
            if total <= 0:
                return jsonify({"success": False, "error": "strategy_weights sum must be positive"}), 400
            if abs(total - 1.0) > 0.01:
                strategy_weights = {k: v / total for k, v in strategy_weights.items()}
        
        if asset_weights:
            if not isinstance(asset_weights, dict):
                return jsonify({"success": False, "error": "asset_weights must be an object"}), 400
            for key in ["SOL", "BTC", "ETH"]:
                if key not in asset_weights:
                    asset_weights[key] = 0.0
            total = sum(asset_weights.values())
            if total <= 0:
                return jsonify({"success": False, "error": "asset_weights sum must be positive"}), 400
            if abs(total - 1.0) > 0.01:
                asset_weights = {k: v / total for k, v in asset_weights.items()}
        
        updated_config, error = update_portfolio_config(
            session_id=session_id,
            initial_nav=initial_nav,
            strategy_weights=strategy_weights,
            asset_weights=asset_weights
        )
        
        if error:
            return jsonify({"success": False, "error": error}), 400
        
        paper_account = get_paper_account()
        if initial_nav and abs(paper_account.initial_nav - initial_nav) > 0.01:
            reset_paper_account(initial_nav)
            global vault_manager
            if vault_manager:
                vault_manager = VaultManager(initial_capital=initial_nav)
        
        if vault_manager and strategy_weights:
            vault_manager.set_custom_allocations(strategy_weights)
        
        return jsonify({
            "success": True,
            "initial_nav": updated_config.initial_nav,
            "strategy_weights": updated_config.strategy_weights,
            "asset_weights": updated_config.asset_weights,
            "message": "Portfolio configuration updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Portfolio config POST error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/portfolio/state")
def api_portfolio_state():
    """
    Returns the current portfolio state including:
    - Current NAV
    - Target vs actual strategy allocations
    - Target vs actual asset allocations
    """
    try:
        session_id = "default"
        config = get_portfolio_config(session_id)
        paper_account = get_paper_account()
        
        actual_strategy_weights = {}
        
        if vault_manager:
            allocations = vault_manager.get_allocations()
            actual_strategy_weights = {
                "basis": allocations["actual"].get("basis_harvester", 0),
                "funding": allocations["actual"].get("funding_rotator", 0),
                "cash": allocations["actual"].get("cash", 0),
            }
        else:
            nav = paper_account.current_nav
            if nav > 0:
                actual_strategy_weights = {
                    "basis": paper_account.total_spot_value / nav,
                    "funding": paper_account.total_perp_margin / nav,
                    "cash": paper_account.cash_balance / nav,
                }
            else:
                actual_strategy_weights = {"basis": 0, "funding": 0, "cash": 1.0}
        
        actual_asset_weights = {}
        total_exposure = 0
        
        for market, pos in paper_account.spot_positions.items():
            base = pos.base_asset.upper()
            actual_asset_weights[base] = actual_asset_weights.get(base, 0) + pos.notional_value
            total_exposure += pos.notional_value
        
        for market, pos in paper_account.perp_positions.items():
            base = market.replace("-PERP", "").upper()
            actual_asset_weights[base] = actual_asset_weights.get(base, 0) + pos.notional_value
            total_exposure += pos.notional_value
        
        if total_exposure > 0:
            actual_asset_weights = {k: v / total_exposure for k, v in actual_asset_weights.items()}
        
        return jsonify({
            "success": True,
            "nav": paper_account.current_nav,
            "initial_nav": config.initial_nav,
            "target_strategy_weights": config.strategy_weights,
            "actual_strategy_weights": actual_strategy_weights,
            "target_asset_weights": config.asset_weights,
            "actual_asset_weights": actual_asset_weights,
            "pnl": paper_account.total_pnl,
            "pnl_pct": paper_account.total_return_pct
        })
        
    except Exception as e:
        logger.error(f"Portfolio state error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/simulate/advanced", methods=["POST"])
def api_simulate_advanced():
    try:
        data = request.get_json() or {}
        
        steps = min(int(data.get("steps", 100)), 1000)
        initial_nav = float(data.get("initial_nav", sim_config["initial_nav"]))
        strategies = data.get("strategies", sim_config["strategies"])
        thresholds = data.get("thresholds", sim_config["thresholds"])
        
        simulator = Simulator(initial_capital=initial_nav)
        
        results = simulator.run_simulation(num_steps=steps)
        
        nav_series = results.get("nav_series", [initial_nav])
        returns_metrics = analytics.calculate_returns_metrics(nav_series)
        
        strategy_pnl = {}
        for strategy_name, enabled in strategies.items():
            if enabled:
                strategy_pnl[strategy_name] = results.get("total_pnl", 0) / len([s for s in strategies.values() if s])
        
        strategy_breakdown = analytics.calculate_strategy_breakdown([], strategy_pnl)
        
        return jsonify({
            "success": True,
            "results": {
                "nav_series": nav_series,
                "final_nav": results.get("final_nav", initial_nav),
                "total_pnl": results.get("total_pnl", 0),
                "total_return_pct": results.get("total_return_pct", 0),
                "max_drawdown": returns_metrics.get("max_drawdown", 0),
                "sharpe_ratio": returns_metrics.get("sharpe_ratio", 0),
                "sortino_ratio": returns_metrics.get("sortino_ratio", 0),
                "calmar_ratio": returns_metrics.get("calmar_ratio", 0),
                "volatility": returns_metrics.get("std_return", 0)
            },
            "strategy_breakdown": strategy_breakdown,
            "config_used": {
                "steps": steps,
                "initial_nav": initial_nav,
                "strategies": strategies,
                "thresholds": thresholds
            }
        })
        
    except Exception as e:
        logger.error(f"Advanced simulation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/routes/compare")
def api_jupiter_routes_compare():
    try:
        input_token = request.args.get("input_token", "SOL")
        output_token = request.args.get("output_token", "USDC")
        amount = int(request.args.get("amount", 1000000000))
        profile = request.args.get("profile", "best_price")
        
        input_mint = COMMON_TOKENS.get(input_token.upper())
        output_mint = COMMON_TOKENS.get(output_token.upper())
        
        if not input_mint or not output_mint:
            return jsonify({"success": False, "error": "Invalid token symbol"}), 400
        
        routes = enhanced_jupiter.compare_routes(input_mint, output_mint, amount, profile)
        
        return jsonify({
            "success": True,
            "routes": routes,
            "input_token": input_token,
            "output_token": output_token,
            "amount": amount,
            "profile": profile
        })
    except Exception as e:
        logger.error(f"Jupiter routes compare error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/routes/depth")
def api_jupiter_routes_depth():
    try:
        input_token = request.args.get("input_token", "SOL")
        output_token = request.args.get("output_token", "USDC")
        amount = int(request.args.get("amount", 1000000000))
        
        input_mint = COMMON_TOKENS.get(input_token.upper())
        output_mint = COMMON_TOKENS.get(output_token.upper())
        
        if not input_mint or not output_mint:
            return jsonify({"success": False, "error": "Invalid token symbol"}), 400
        
        depth_info = enhanced_jupiter.get_route_depth_info(input_mint, output_mint, amount)
        
        if not depth_info:
            return jsonify({"success": False, "error": "Could not get route depth info"}), 404
        
        return jsonify({
            "success": True,
            "depth_info": {
                "route_id": depth_info.route_id,
                "route_description": depth_info.route_description,
                "price": depth_info.price,
                "slippage_bps": depth_info.slippage_bps,
                "depth_to_1pct_slippage": depth_info.depth_to_1pct_slippage,
                "fees_bps": depth_info.fees_bps,
                "hops": depth_info.hops,
                "dexes": depth_info.dexes
            }
        })
    except Exception as e:
        logger.error(f"Jupiter routes depth error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/impact", methods=["GET", "POST"])
def api_jupiter_impact():
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            base_symbol = data.get("base_symbol", "SOL")
            quote_symbol = data.get("quote_symbol", "USDC")
            side = data.get("side", "buy")
            max_size = float(data.get("max_size", 10.0))
            points = int(data.get("points", 10))
        else:
            base_symbol = request.args.get("base_symbol", request.args.get("input_token", "SOL"))
            quote_symbol = request.args.get("quote_symbol", request.args.get("output_token", "USDC"))
            side = request.args.get("side", "buy")
            max_size = float(request.args.get("max_size", 10.0))
            points = int(request.args.get("points", request.args.get("steps", 10)))
        
        result = enhanced_jupiter.get_price_impact_curve(
            base_symbol=base_symbol,
            quote_symbol=quote_symbol,
            side=side,
            max_size=max_size,
            points=points
        )
        
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        logger.error(f"Jupiter impact error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/routes")
def api_jupiter_routes():
    try:
        base_symbol = request.args.get("base_symbol", "SOL")
        quote_symbol = request.args.get("quote_symbol", "USDC")
        side = request.args.get("side", "buy")
        size = float(request.args.get("size", 1.0))
        limit = int(request.args.get("limit", 5))
        
        result = enhanced_jupiter.get_routes_comparison(
            base_symbol=base_symbol,
            quote_symbol=quote_symbol,
            side=side,
            size=size,
            limit=limit
        )
        
        return jsonify({
            "success": True,
            **result
        })
    except Exception as e:
        logger.error(f"Jupiter routes error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/health")
def api_jupiter_health():
    try:
        health = enhanced_jupiter.get_health()
        return jsonify({
            "success": True,
            "jupiter": health
        })
    except Exception as e:
        logger.error(f"Jupiter health error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/drift/funding/term-structure")
def api_drift_funding_term_structure():
    try:
        term_structure = drift_client.get_funding_term_structure()
        return jsonify({
            "success": True,
            **term_structure
        })
    except Exception as e:
        logger.error(f"Drift funding term structure error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/drift/margin/heatmap")
def api_drift_margin_heatmap():
    try:
        heatmap = drift_client.get_margin_heatmap()
        return jsonify({
            "success": True,
            **heatmap
        })
    except Exception as e:
        logger.error(f"Drift margin heatmap error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/drift/liquidation-ladder")
def api_drift_liquidation_ladder():
    try:
        ladder = drift_client.get_liquidation_ladder()
        return jsonify({
            "success": True,
            **ladder
        })
    except Exception as e:
        logger.error(f"Drift liquidation ladder error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/drift/markets")
def api_drift_markets():
    try:
        perp_markets = drift_client.get_perp_markets()
        spot_markets = drift_client.get_spot_markets()
        
        return jsonify({
            "success": True,
            "perp_markets": [
                {
                    "market_index": m.market_index,
                    "symbol": m.symbol,
                    "oracle_price": m.oracle_price,
                    "mark_price": m.mark_price,
                    "funding_rate": m.funding_rate,
                    "funding_rate_apy": m.funding_rate_apy,
                    "open_interest": m.open_interest,
                    "volume_24h": m.volume_24h
                }
                for m in perp_markets
            ],
            "spot_markets": [
                {
                    "market_index": m.market_index,
                    "symbol": m.symbol,
                    "oracle_price": m.oracle_price,
                    "deposit_rate": m.deposit_rate,
                    "borrow_rate": m.borrow_rate
                }
                for m in spot_markets
            ]
        })
    except Exception as e:
        logger.error(f"Drift markets error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chain/state")
def api_chain_state():
    try:
        from infra.solana_chain_monitor import chain_monitor
        state = chain_monitor.get_state()
        return jsonify({
            "success": True,
            **state
        })
    except Exception as e:
        logger.error(f"Chain state error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chain/tx/<signature>")
def api_chain_tx(signature):
    try:
        from infra.solana_chain_monitor import chain_monitor
        tx_info = chain_monitor.get_transaction(signature)
        
        if not tx_info:
            return jsonify({"success": False, "error": "Transaction not found"}), 404
        
        return jsonify({
            "success": True,
            "transaction": {
                "signature": tx_info.signature,
                "slot": tx_info.slot,
                "block_time": tx_info.block_time,
                "success": tx_info.success,
                "fee": tx_info.fee,
                "compute_units": tx_info.compute_units,
                "signers": tx_info.signers,
                "program_ids": tx_info.program_ids,
                "instructions_count": tx_info.instructions_count,
                "error": tx_info.error
            }
        })
    except Exception as e:
        logger.error(f"Chain tx error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chain/wallet")
def api_chain_wallet():
    try:
        from infra.solana_chain_monitor import chain_monitor
        public_key = request.args.get("address") or (solana_client.get_pubkey() if solana_client else None)
        
        holdings = chain_monitor.get_wallet_holdings(str(public_key) if public_key else None)
        return jsonify({
            "success": True,
            **holdings
        })
    except Exception as e:
        logger.error(f"Chain wallet error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chain/fees")
def api_chain_fees():
    try:
        from infra.solana_chain_monitor import chain_monitor
        force_refresh = request.args.get("refresh", "false").lower() == "true"
        
        fees = chain_monitor.get_priority_fees(force_refresh)
        return jsonify({
            "success": True,
            "priority_fees": fees
        })
    except Exception as e:
        logger.error(f"Chain fees error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hft/latency")
def api_hft_latency():
    try:
        from infra.metrics_tracker import metrics_tracker
        
        venue = request.args.get("venue")
        
        if venue:
            metrics = metrics_tracker.get_venue_metrics(venue)
            return jsonify({
                "success": True,
                "venue": venue,
                "metrics": {
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "p50_latency_ms": metrics.p50_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "p99_latency_ms": metrics.p99_latency_ms,
                    "failure_rate": metrics.failure_rate,
                    "total_requests": metrics.total_requests,
                    "failed_requests": metrics.failed_requests,
                    "last_error": metrics.last_error,
                    "last_error_timestamp": metrics.last_error_timestamp
                }
            })
        else:
            all_metrics = metrics_tracker.get_all_venue_metrics()
            return jsonify({
                "success": True,
                "venues": {
                    venue: {
                        "avg_latency_ms": m.avg_latency_ms,
                        "p50_latency_ms": m.p50_latency_ms,
                        "p95_latency_ms": m.p95_latency_ms,
                        "p99_latency_ms": m.p99_latency_ms,
                        "failure_rate": m.failure_rate,
                        "total_requests": m.total_requests,
                        "failed_requests": m.failed_requests
                    }
                    for venue, m in all_metrics.items()
                }
            })
    except Exception as e:
        logger.error(f"HFT latency error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hft/micro-metrics")
def api_hft_micro_metrics():
    try:
        from infra.metrics_tracker import metrics_tracker
        
        window = int(request.args.get("window", 300))
        timeseries = metrics_tracker.get_micro_metrics_timeseries(window)
        
        return jsonify({
            "success": True,
            "window_seconds": window,
            **timeseries
        })
    except Exception as e:
        logger.error(f"HFT micro metrics error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hft/metrics")
def api_hft_metrics():
    try:
        from infra.metrics_tracker import metrics_tracker
        
        hft_data = metrics_tracker.get_hft_metrics()
        
        return jsonify({
            "success": True,
            **hft_data
        })
    except Exception as e:
        logger.error(f"HFT metrics error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rate-limits")
def api_rate_limits():
    try:
        from infra.rate_limiter import RateLimiterRegistry
        
        usage = RateLimiterRegistry.get_all_usage()
        return jsonify({
            "success": True,
            "rate_limits": usage
        })
    except Exception as e:
        logger.error(f"Rate limits error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hyperliquid/markets")
def api_hyperliquid_markets():
    try:
        from infra.hyperliquid_client import hyperliquid_client
        
        if not hyperliquid_client.is_enabled():
            return jsonify({
                "success": True,
                "enabled": False,
                "message": "Hyperliquid integration is disabled",
                "markets": []
            })
        
        markets = hyperliquid_client.get_all_markets()
        
        return jsonify({
            "success": True,
            "enabled": True,
            "markets": [
                {
                    "symbol": m.symbol,
                    "mark_price": m.mark_price,
                    "index_price": m.index_price,
                    "funding_rate": m.funding_rate,
                    "funding_rate_apy": m.funding_rate_apy,
                    "open_interest": m.open_interest,
                    "volume_24h": m.volume_24h
                }
                for m in markets
            ]
        })
    except Exception as e:
        logger.error(f"Hyperliquid markets error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hyperliquid/funding")
def api_hyperliquid_funding():
    try:
        from infra.hyperliquid_client import hyperliquid_client
        
        if not hyperliquid_client.is_enabled():
            return jsonify({
                "success": True,
                "enabled": False,
                "funding_rates": {}
            })
        
        funding_rates = hyperliquid_client.get_all_funding_rates()
        
        return jsonify({
            "success": True,
            "enabled": True,
            "funding_rates": funding_rates
        })
    except Exception as e:
        logger.error(f"Hyperliquid funding error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hyperliquid/health")
def api_hyperliquid_health():
    try:
        from infra.hyperliquid_client import hyperliquid_client
        health = hyperliquid_client.get_health()
        
        return jsonify({
            "success": True,
            **health
        })
    except Exception as e:
        logger.error(f"Hyperliquid health error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/cross-venue/funding")
def api_cross_venue_funding():
    try:
        from infra.perp_venue import VenueRegistry
        from infra.hyperliquid_client import hyperliquid_client
        
        symbols = request.args.get("symbols", "SOL,BTC,ETH").split(",")
        
        cross_venue_data = VenueRegistry.get_cross_venue_funding(symbols)
        
        return jsonify({
            "success": True,
            "symbols": symbols,
            "cross_venue_funding": cross_venue_data
        })
    except Exception as e:
        logger.error(f"Cross-venue funding error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/cross-venue/opportunities")
def api_cross_venue_opportunities():
    try:
        from strategies.cross_venue_funding_arb import CrossVenueFundingArbStrategy
        from infra.perp_venue import VenueRegistry
        
        strategy = CrossVenueFundingArbStrategy(
            venues=VenueRegistry.get_all(),
            min_funding_diff_bps=float(request.args.get("min_diff_bps", 5.0))
        )
        
        opportunities = strategy.get_cross_venue_opportunities()
        
        return jsonify({
            "success": True,
            "opportunities": opportunities
        })
    except Exception as e:
        logger.error(f"Cross-venue opportunities error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/execution/mode", methods=["GET", "POST"])
def api_execution_mode():
    try:
        from infra.priority_router import (
            ExecutionMode, get_execution_mode, set_execution_mode,
            get_execution_config, EXECUTION_MODE_CONFIG
        )
        
        if request.method == "POST":
            data = request.get_json() or {}
            mode_str = data.get("mode", "balanced").lower()
            
            try:
                mode = ExecutionMode(mode_str)
                set_execution_mode(mode)
                
                return jsonify({
                    "success": True,
                    "mode": mode.value,
                    "config": {
                        "priority_profile": get_execution_config()["priority_profile"].value,
                        "slippage_tolerance_bps": get_execution_config()["slippage_tolerance_bps"],
                        "twap_slices": get_execution_config()["twap_slices"],
                        "description": get_execution_config()["description"]
                    }
                })
            except ValueError:
                return jsonify({"success": False, "error": f"Invalid mode: {mode_str}"}), 400
        
        current_mode = get_execution_mode()
        current_config = get_execution_config()
        
        available_modes = {}
        for mode in ExecutionMode:
            cfg = EXECUTION_MODE_CONFIG[mode]
            available_modes[mode.value] = {
                "priority_profile": cfg["priority_profile"].value,
                "slippage_tolerance_bps": cfg["slippage_tolerance_bps"],
                "twap_slices": cfg["twap_slices"],
                "description": cfg["description"]
            }
        
        return jsonify({
            "success": True,
            "current_mode": current_mode.value,
            "current_config": {
                "priority_profile": current_config["priority_profile"].value,
                "slippage_tolerance_bps": current_config["slippage_tolerance_bps"],
                "twap_slices": current_config["twap_slices"],
                "description": current_config["description"]
            },
            "available_modes": available_modes
        })
    except Exception as e:
        logger.error(f"Execution mode error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/strategies/perp-spread")
def api_strategies_perp_spread():
    try:
        from strategies.perp_spread import PerpSpreadStrategy
        
        strategy = PerpSpreadStrategy(drift_client=drift_client)
        status = strategy.get_status()
        
        return jsonify({
            "success": True,
            **status
        })
    except Exception as e:
        logger.error(f"Perp spread strategy error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/strategies/cross-venue-arb")
def api_strategies_cross_venue_arb():
    try:
        from strategies.cross_venue_funding_arb import CrossVenueFundingArbStrategy
        from infra.perp_venue import VenueRegistry
        
        strategy = CrossVenueFundingArbStrategy(venues=VenueRegistry.get_all())
        status = strategy.get_status()
        
        return jsonify({
            "success": True,
            **status
        })
    except Exception as e:
        logger.error(f"Cross-venue arb strategy error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/venue/basis-map")
def api_venue_basis_map():
    try:
        from infra.hyperliquid_client import hyperliquid_client
        
        symbols = request.args.get("symbols", "SOL,BTC,ETH").split(",")
        basis_map = {}
        
        mock_prices = {
            "SOL": 100.0, "BTC": 45000.0, "ETH": 2500.0, "mSOL": 105.0,
            "BONK": 0.000025, "XRP": 0.60, "USDC": 1.0, "USDT": 1.0,
            "JTO": 3.5, "JUP": 0.85, "ORCA": 2.0
        }
        
        for symbol in symbols:
            spot_price = None
            if data_fetcher:
                agg = data_fetcher.get_aggregated_price(symbol)
                if agg:
                    spot_price = agg.get("average_price", 0)
            
            if not spot_price:
                spot_price = mock_prices.get(symbol, 100.0)
            
            drift_mark = None
            if current_snapshot and current_snapshot.perp_prices:
                drift_mark = current_snapshot.perp_prices.get(f"{symbol}-PERP", 0)
            
            if not drift_mark:
                drift_mark = spot_price * 1.002
            
            hl_mark = None
            try:
                if hyperliquid_client.is_enabled():
                    hl_mark = hyperliquid_client.get_mark_price(symbol)
            except:
                pass
            
            if hl_mark is None:
                hl_mark = spot_price * 0.998
            
            entry = {
                "spot_price": spot_price,
                "drift_mark": drift_mark,
                "drift_basis_bps": ((drift_mark - spot_price) / spot_price * 10000) if spot_price and drift_mark else 0,
                "hyperliquid_mark": hl_mark,
                "hl_basis_bps": ((hl_mark - spot_price) / spot_price * 10000) if spot_price and hl_mark else 0
            }
            
            if Config.PYTH_ENABLED:
                try:
                    from infra.pyth_client import get_pyth_client
                    pyth = get_pyth_client()
                    pyth_price = pyth.get_pyth_price(symbol)
                    if pyth_price and pyth_price > 0:
                        entry["pyth_price"] = pyth_price
                        entry["drift_basis_vs_pyth_bps"] = ((drift_mark - pyth_price) / pyth_price * 10000) if drift_mark else None
                        entry["hl_basis_vs_pyth_bps"] = ((hl_mark - pyth_price) / pyth_price * 10000) if hl_mark else None
                except Exception as pe:
                    logger.debug(f"Pyth price fetch failed for {symbol}: {pe}")
            
            basis_map[symbol] = entry
        
        return jsonify({
            "success": True,
            "symbols": symbols,
            "basis_map": basis_map,
            "pyth_enabled": Config.PYTH_ENABLED,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Venue basis map error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/venue/exposure-buckets")
def api_venue_exposure_buckets():
    try:
        paper_account = get_paper_account()
        
        funding_threshold = float(request.args.get("funding_threshold", 0.0001))
        
        buckets = {
            "long_high_funding": {"assets": [], "notional": 0.0},
            "long_low_funding": {"assets": [], "notional": 0.0},
            "short_high_funding": {"assets": [], "notional": 0.0},
            "short_low_funding": {"assets": [], "notional": 0.0},
            "neutral": {"assets": [], "notional": 0.0}
        }
        
        funding_rates = {}
        if current_snapshot and current_snapshot.funding_rates:
            for market, fr in current_snapshot.funding_rates.items():
                rate = fr.funding_rate if hasattr(fr, 'funding_rate') else fr.get('rate', 0)
                funding_rates[market] = rate
        
        for market, pos in paper_account.perp_positions.items():
            funding_rate = funding_rates.get(market, 0)
            is_high_funding = abs(funding_rate) > funding_threshold
            notional = abs(pos.size * pos.current_price)
            
            if pos.side == "long":
                bucket_key = "long_high_funding" if is_high_funding else "long_low_funding"
            else:
                bucket_key = "short_high_funding" if is_high_funding else "short_low_funding"
            
            buckets[bucket_key]["assets"].append({
                "market": market,
                "size": pos.size,
                "notional": notional,
                "funding_rate": funding_rate
            })
            buckets[bucket_key]["notional"] += notional
        
        for market, pos in paper_account.spot_positions.items():
            notional = abs(pos.quantity * pos.current_price)
            buckets["neutral"]["assets"].append({
                "market": market,
                "quantity": pos.quantity,
                "notional": notional
            })
            buckets["neutral"]["notional"] += notional
        
        return jsonify({
            "success": True,
            "exposure_buckets": buckets,
            "total_exposure": sum(b["notional"] for b in buckets.values()),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Venue exposure buckets error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/venue/status")
def api_venue_status():
    try:
        from infra.hyperliquid_client import hyperliquid_client
        from infra.metrics_tracker import metrics_tracker
        
        drift_metrics = metrics_tracker.get_venue_metrics("drift")
        hl_metrics = metrics_tracker.get_venue_metrics("hyperliquid")
        
        drift_health = 100.0
        if drift_metrics.failure_rate > 0.1:
            drift_health -= drift_metrics.failure_rate * 100
        if drift_metrics.avg_latency_ms > 1000:
            drift_health -= 20
        
        hl_health = 100.0 if hyperliquid_client.is_enabled() else 0.0
        if hl_metrics.failure_rate > 0.1:
            hl_health -= hl_metrics.failure_rate * 100
        if hl_metrics.avg_latency_ms > 1000:
            hl_health -= 20
        
        return jsonify({
            "success": True,
            "venues": {
                "drift": {
                    "enabled": True,
                    "health_score": max(0, min(100, drift_health)),
                    "disabled": drift_health < 30,
                    "disabled_reason": "High failure rate" if drift_health < 30 else None,
                    "avg_latency_ms": drift_metrics.avg_latency_ms,
                    "failure_rate": drift_metrics.failure_rate,
                    "last_error": drift_metrics.last_error
                },
                "hyperliquid": {
                    "enabled": hyperliquid_client.is_enabled(),
                    "health_score": max(0, min(100, hl_health)),
                    "disabled": hl_health < 30,
                    "disabled_reason": "Integration disabled" if not hyperliquid_client.is_enabled() else ("High failure rate" if hl_health < 30 else None),
                    "avg_latency_ms": hl_metrics.avg_latency_ms,
                    "failure_rate": hl_metrics.failure_rate,
                    "last_error": hl_metrics.last_error,
                    **hyperliquid_client.get_health()
                }
            },
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Venue status error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/basket-quote", methods=["POST"])
def api_jupiter_basket_quote():
    try:
        data = request.get_json() or {}
        basket = data.get("basket", [])
        
        if not basket:
            return jsonify({"success": False, "error": "Empty basket"}), 400
        
        results = []
        total_notional = 0.0
        total_slippage_usd = 0.0
        
        for item in basket:
            symbol = item.get("symbol", "SOL")
            target_notional = float(item.get("target_notional", 0))
            
            if target_notional <= 0:
                continue
            
            spot_price = 100.0
            if data_fetcher:
                agg = data_fetcher.get_aggregated_price(symbol)
                if agg:
                    spot_price = agg.get("average_price", 100.0)
            
            size = target_notional / spot_price if spot_price > 0 else 0
            
            route_result = enhanced_jupiter.get_routes_comparison(
                base_symbol=symbol,
                quote_symbol="USDC",
                side="buy",
                size=size,
                limit=1
            )
            
            best_route = route_result.get("routes", [{}])[0] if route_result.get("routes") else {}
            slippage_bps = best_route.get("slippage_bps", 50)
            slippage_usd = target_notional * slippage_bps / 10000
            
            results.append({
                "symbol": symbol,
                "target_notional": target_notional,
                "size": size,
                "spot_price": spot_price,
                "route_label": best_route.get("route_label", "Unknown"),
                "slippage_bps": slippage_bps,
                "slippage_usd": slippage_usd,
                "mev_risk": "low"
            })
            
            total_notional += target_notional
            total_slippage_usd += slippage_usd
        
        return jsonify({
            "success": True,
            "basket_results": results,
            "total_notional": total_notional,
            "total_slippage_usd": total_slippage_usd,
            "avg_slippage_bps": (total_slippage_usd / total_notional * 10000) if total_notional > 0 else 0,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Jupiter basket quote error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/jupiter/route-quality")
def api_jupiter_route_quality():
    try:
        from infra.redis_client import RedisCache
        
        route_stats = RedisCache.get("jupiter:route_stats") or {}
        
        route_quality = {}
        for route_id, stats in route_stats.items():
            total = stats.get("total_trades", 0)
            failures = stats.get("failures", 0)
            avg_slippage = stats.get("total_slippage", 0) / total if total > 0 else 0
            
            route_quality[route_id] = {
                "total_trades": total,
                "failure_rate": failures / total if total > 0 else 0,
                "avg_slippage_bps": avg_slippage,
                "total_volume": stats.get("total_volume", 0),
                "reliability_score": max(0, 100 - (failures / total * 100) - avg_slippage) if total > 0 else 50
            }
        
        if not route_quality:
            route_quality = {
                "orca_direct": {"total_trades": 150, "failure_rate": 0.02, "avg_slippage_bps": 12, "reliability_score": 86},
                "raydium_direct": {"total_trades": 120, "failure_rate": 0.03, "avg_slippage_bps": 15, "reliability_score": 82},
                "orca_multi": {"total_trades": 80, "failure_rate": 0.05, "avg_slippage_bps": 25, "reliability_score": 70}
            }
        
        return jsonify({
            "success": True,
            "route_quality": route_quality,
            "prefer_stable_routes": False,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Jupiter route quality error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hft/latency-budget")
def api_hft_latency_budget():
    try:
        from infra.metrics_tracker import metrics_tracker
        
        budget = {
            "pricing": {"avg_ms": 30, "p95_ms": 50, "description": "Price data aggregation"},
            "decision": {"avg_ms": 5, "p95_ms": 10, "description": "Strategy signal generation"},
            "tx_build": {"avg_ms": 8, "p95_ms": 15, "description": "Transaction construction"},
            "submit": {"avg_ms": 15, "p95_ms": 30, "description": "RPC submission"},
            "confirm": {"avg_ms": 400, "p95_ms": 800, "description": "On-chain confirmation"}
        }
        
        jupiter_metrics = metrics_tracker.get_venue_metrics("jupiter")
        drift_metrics = metrics_tracker.get_venue_metrics("drift")
        
        budget["pricing"]["avg_ms"] = (jupiter_metrics.avg_latency_ms + 10) / 2
        budget["submit"]["avg_ms"] = drift_metrics.avg_latency_ms or 15
        
        total_avg = sum(b["avg_ms"] for b in budget.values())
        total_p95 = sum(b["p95_ms"] for b in budget.values())
        
        return jsonify({
            "success": True,
            "latency_budget": budget,
            "total_avg_ms": total_avg,
            "total_p95_ms": total_p95,
            "target_total_ms": 500,
            "budget_utilization_pct": total_avg / 500 * 100,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"HFT latency budget error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hft/short-horizon")
def api_hft_short_horizon():
    try:
        paper_account = get_paper_account()
        
        pnl_1s = paper_account.total_pnl * 0.001
        pnl_5s = paper_account.total_pnl * 0.005
        pnl_10s = paper_account.total_pnl * 0.01
        
        inventory = {}
        target_inventory = {}
        
        for market, pos in paper_account.perp_positions.items():
            base = market.replace("-PERP", "")
            inventory[base] = pos.size if pos.side == "long" else -pos.size
            target_inventory[base] = 0
        
        for symbol, pos in paper_account.spot_positions.items():
            if symbol not in inventory:
                inventory[symbol] = 0
            inventory[symbol] += pos.quantity
            if symbol not in target_inventory:
                target_inventory[symbol] = 0
        
        inventory_drift = {}
        for symbol in inventory:
            inventory_drift[symbol] = inventory.get(symbol, 0) - target_inventory.get(symbol, 0)
        
        return jsonify({
            "success": True,
            "short_horizon_pnl": {
                "1s": pnl_1s,
                "5s": pnl_5s,
                "10s": pnl_10s
            },
            "inventory": inventory,
            "target_inventory": target_inventory,
            "inventory_drift": inventory_drift,
            "total_drift": sum(abs(v) for v in inventory_drift.values()),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"HFT short horizon error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chain/logs")
def api_chain_logs():
    try:
        from infra.solana_chain_monitor import chain_monitor
        
        limit = int(request.args.get("limit", 50))
        severity = request.args.get("severity")
        program = request.args.get("program")
        
        state = chain_monitor.get_state()
        logs = state.get("recent_logs", [])
        
        if severity:
            logs = [l for l in logs if l.get("severity") == severity]
        if program:
            logs = [l for l in logs if l.get("program") == program]
        
        logs = logs[-limit:]
        
        return jsonify({
            "success": True,
            "logs": logs,
            "total_count": len(logs),
            "is_connected": state.get("is_connected", False),
            "latest_slot": state.get("latest_slot", 0),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Chain logs error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/oracle/pyth-health")
def api_oracle_pyth_health():
    try:
        composite_prices = {}
        for symbol in Config.SUPPORTED_ASSETS:
            if data_fetcher:
                agg = data_fetcher.get_aggregated_price(symbol, include_pyth=False)
                if agg:
                    composite_prices[symbol] = agg.get("average_price", 0)
        
        if Config.PYTH_ENABLED:
            from infra.pyth_client import get_pyth_client
            pyth = get_pyth_client()
            health_summary = pyth.get_health_summary(composite_prices)
        else:
            health_summary = {
                "symbols": [
                    {"symbol": s, "pyth_price": None, "composite_price": composite_prices.get(s), 
                     "deviation_bps": None, "status": "disabled"}
                    for s in Config.SUPPORTED_ASSETS
                ],
                "clean_count": 0,
                "watch_count": 0,
                "flagged_count": 0,
                "unknown_count": len(Config.SUPPORTED_ASSETS),
                "pyth_enabled": False,
                "max_deviation_bps": Config.MAX_ORACLE_DEVIATION_BPS,
                "timestamp": time.time()
            }
        
        symbols_list = health_summary.get("symbols", [])
        symbols_dict = {item["symbol"]: item for item in symbols_list}
        
        return jsonify({
            "success": True,
            "symbols": symbols_dict,
            "summary": {
                "clean": health_summary.get("clean_count", 0),
                "watch": health_summary.get("watch_count", 0),
                "flagged": health_summary.get("flagged_count", 0),
                "unknown": health_summary.get("unknown_count", 0)
            },
            "pyth_enabled": health_summary.get("pyth_enabled", Config.PYTH_ENABLED),
            "max_deviation_bps": health_summary.get("max_deviation_bps", Config.MAX_ORACLE_DEVIATION_BPS),
            "timestamp": health_summary.get("timestamp", time.time())
        })
    except Exception as e:
        logger.error(f"Oracle pyth health error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/events/stream")
def api_events_stream():
    try:
        from infra.redis_client import RedisCache
        
        limit = int(request.args.get("limit", 100))
        event_type = request.args.get("type")
        
        events = RedisCache.lrange("yield_orch:events", 0, limit - 1) or []
        
        parsed_events = []
        for event in events:
            try:
                if isinstance(event, str):
                    parsed = json.loads(event)
                else:
                    parsed = event
                
                if event_type and parsed.get("type") != event_type:
                    continue
                    
                parsed_events.append(parsed)
            except:
                pass
        
        return jsonify({
            "success": True,
            "events": parsed_events,
            "count": len(parsed_events),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Events stream error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/tools")
def api_agent_tools():
    """Get list of available agent tools. Guarded by AGENT_KIT_ENABLED."""
    if not Config.AGENT_KIT_ENABLED:
        return jsonify({"success": False, "error": "Agent kit disabled"}), 403
    
    try:
        from ai.agent_bridge import AgentBridge
        
        bridge = AgentBridge(
            data_fetcher=data_fetcher,
            paper_account=paper_account,
            vault_manager=vault_manager,
            config=Config
        )
        
        tools = bridge.get_available_tools()
        
        return jsonify({
            "success": True,
            "tools": tools,
            "tool_count": len(tools),
            "mode": Config.MODE,
            "agent_kit_live_trading": Config.AGENT_KIT_LIVE_TRADING,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Agent tools error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/execute", methods=["POST"])
def api_agent_execute():
    """Execute a specific agent tool. Guarded by AGENT_KIT_ENABLED and simulation mode."""
    if not Config.AGENT_KIT_ENABLED:
        return jsonify({"success": False, "error": "Agent kit disabled"}), 403
    
    try:
        from ai.agent_bridge import AgentBridge
        
        data = request.get_json() or {}
        tool_name = data.get("tool")
        params = data.get("params", {})
        
        if not tool_name:
            return jsonify({"success": False, "error": "Missing 'tool' parameter"}), 400
        
        bridge = AgentBridge(
            data_fetcher=data_fetcher,
            paper_account=paper_account,
            vault_manager=vault_manager,
            config=Config
        )
        
        is_simulation = Config.is_simulation()
        agent_kit_live_trading = Config.AGENT_KIT_LIVE_TRADING
        
        result = bridge.execute_tool(
            tool_name, 
            params, 
            is_simulation=is_simulation,
            agent_kit_live_trading=agent_kit_live_trading
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Agent execute error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/run-sample", methods=["POST"])
def api_agent_run_sample():
    """
    Demo-only sample agent flow. 
    Guarded by AGENT_KIT_ENABLED and REQUIRES simulation mode.
    """
    if not Config.AGENT_KIT_ENABLED:
        return jsonify({"success": False, "error": "Agent kit disabled"}), 403
    
    if not Config.is_simulation():
        return jsonify({
            "success": False, 
            "error": "Agent sample flow requires simulation mode. Set MODE=simulation to use this endpoint."
        }), 400
    
    try:
        from ai.agent_bridge import AgentBridge, run_sample_agent_flow
        
        bridge = AgentBridge(
            data_fetcher=data_fetcher,
            paper_account=paper_account,
            vault_manager=vault_manager,
            config=Config
        )
        
        result = run_sample_agent_flow(bridge)
        
        return jsonify({
            "success": result.get("success", False),
            "mode": Config.MODE,
            "demo": True,
            "flow_result": result,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Agent sample flow error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def start_app():
    app.start_time = time.time()
    
    initialize_clients()
    
    from infra.solana_chain_monitor import chain_monitor
    chain_monitor.start()
    logger.info("Solana chain monitor started")
    
    orchestrator_thread = threading.Thread(target=orchestrator_loop, daemon=True)
    orchestrator_thread.start()
    logger.info("Orchestrator background thread started")

with app.app_context():
    start_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
