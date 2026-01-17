import time
import math
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

MOCK_BASE_PRICES = {
    "SOL": 125.50,
    "mSOL": 145.75,
    "BTC": 97500.00,
    "ETH": 3650.00,
    "USDC": 1.00,
    "USDT": 1.00,
    "JTO": 3.25,
    "JUP": 1.15,
    "BONK": 0.000025,
    "XRP": 2.35,
    "WIF": 2.85,
    "PYTH": 0.45,
    "RAY": 5.20,
    "ORCA": 4.15,
    "MNDE": 0.12,
    "HNT": 6.80,
    "RENDER": 8.45,
    "APT": 12.75,
    "ARB": 1.05,
    "OP": 2.35,
    "SUI": 4.20,
}

MOCK_PERP_MARKETS = {
    "SOL-PERP": {
        "base_price": 125.50,
        "funding_rate": 0.0001,
        "market_index": 0,
    },
    "BTC-PERP": {
        "base_price": 97500.00,
        "funding_rate": 0.00008,
        "market_index": 1,
    },
    "ETH-PERP": {
        "base_price": 3650.00,
        "funding_rate": 0.00006,
        "market_index": 2,
    },
    "APT-PERP": {
        "base_price": 12.75,
        "funding_rate": 0.00015,
        "market_index": 3,
    },
    "ARB-PERP": {
        "base_price": 1.05,
        "funding_rate": -0.0001,
        "market_index": 4,
    },
    "SUI-PERP": {
        "base_price": 4.20,
        "funding_rate": 0.00012,
        "market_index": 5,
    },
    "OP-PERP": {
        "base_price": 2.35,
        "funding_rate": 0.00007,
        "market_index": 6,
    },
    "JTO-PERP": {
        "base_price": 3.25,
        "funding_rate": 0.00009,
        "market_index": 7,
    },
}


def _deterministic_noise(base_price: float, symbol: str, volatility: float = 0.001) -> float:
    hash_input = f"{symbol}_{int(time.time() // 60)}"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
    normalized = (hash_val / 0xFFFFFFFF) * 2 - 1
    noise = normalized * volatility
    return base_price * (1 + noise)


def _get_time_based_variation(base_price: float, amplitude: float = 0.02) -> float:
    t = time.time()
    wave = math.sin(t / 60) * amplitude + math.sin(t / 300) * (amplitude / 2)
    return base_price * (1 + wave)


def _deterministic_value(symbol: str, key: str, min_val: float, max_val: float) -> float:
    hash_input = f"{symbol}_{key}_{int(time.time() // 300)}"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
    normalized = hash_val / 0xFFFFFFFF
    return min_val + normalized * (max_val - min_val)


def get_mock_prices(
    symbols: Optional[List[str]] = None,
    add_noise: bool = True,
    time_variation: bool = True
) -> Dict[str, Dict[str, Any]]:
    if symbols is None:
        symbols = list(MOCK_BASE_PRICES.keys())
    
    prices = {}
    for symbol in symbols:
        if symbol not in MOCK_BASE_PRICES:
            continue
        
        base_price = MOCK_BASE_PRICES[symbol]
        
        if time_variation:
            price = _get_time_based_variation(base_price)
        else:
            price = base_price
        
        if add_noise:
            price = _deterministic_noise(price, symbol)
        
        change_24h = _deterministic_value(symbol, "change", -5, 5)
        volume_24h = base_price * _deterministic_value(symbol, "volume", 1000000, 50000000)
        
        prices[symbol] = {
            "symbol": symbol,
            "price": round(price, 8),
            "source": "mock",
            "change_24h": round(change_24h, 2),
            "volume_24h": round(volume_24h, 2),
            "timestamp": time.time(),
        }
    
    return prices


def get_mock_perp_data(
    markets: Optional[List[str]] = None,
    add_noise: bool = True,
    time_variation: bool = True
) -> Dict[str, Dict[str, Any]]:
    if markets is None:
        markets = list(MOCK_PERP_MARKETS.keys())
    
    perp_data = {}
    for market in markets:
        if market not in MOCK_PERP_MARKETS:
            continue
        
        config = MOCK_PERP_MARKETS[market]
        base_price = config["base_price"]
        
        if time_variation:
            mark_price = _get_time_based_variation(base_price, amplitude=0.015)
        else:
            mark_price = base_price
        
        if add_noise:
            mark_price = _deterministic_noise(mark_price, market, volatility=0.002)
        
        basis_bps = _deterministic_value(market, "basis", -50, 100)
        index_price = mark_price / (1 + basis_bps / 10000)
        
        base_funding = config["funding_rate"]
        funding_noise = _deterministic_value(market, "funding_noise", -0.00005, 0.00005)
        funding_rate = base_funding + funding_noise
        
        funding_apy = funding_rate * 24 * 365 * 100
        
        open_interest = _deterministic_value(market, "oi", 1000000, 50000000)
        
        perp_data[market] = {
            "symbol": market,
            "market_index": config["market_index"],
            "mark_price": round(mark_price, 6),
            "index_price": round(index_price, 6),
            "oracle_price": round(index_price, 6),
            "funding_rate": round(funding_rate, 8),
            "funding_apy": round(funding_apy, 2),
            "basis_bps": round(basis_bps, 2),
            "open_interest": round(open_interest, 2),
            "next_funding_time": time.time() + 3600,
            "timestamp": time.time(),
        }
    
    return perp_data


def _deterministic_hash(seed_str: str) -> float:
    hash_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    return (hash_val / 0xFFFFFFFF) * 2 - 1


def generate_price_series(
    symbol: str,
    length: int = 100,
    interval_seconds: int = 60,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    base_price = MOCK_BASE_PRICES.get(symbol, 100.0)
    
    volatility = 0.02
    drift = 0.0001
    
    series = []
    current_time = time.time() - (length * interval_seconds)
    current_price = base_price
    
    seed_base = seed if seed is not None else 42
    
    for i in range(length):
        returns = _deterministic_hash(f"{symbol}_{seed_base}_{i}_return") * volatility + drift
        current_price *= (1 + returns)
        
        high_noise = abs(_deterministic_hash(f"{symbol}_{seed_base}_{i}_high") * volatility / 2)
        low_noise = abs(_deterministic_hash(f"{symbol}_{seed_base}_{i}_low") * volatility / 2)
        open_noise = _deterministic_hash(f"{symbol}_{seed_base}_{i}_open") * volatility / 3
        
        high = current_price * (1 + high_noise)
        low = current_price * (1 - low_noise)
        open_price = current_price * (1 + open_noise)
        
        vol_hash = int(hashlib.md5(f"{symbol}_{seed_base}_{i}_vol".encode()).hexdigest()[:8], 16)
        volume = 100000 + (vol_hash / 0xFFFFFFFF) * (10000000 - 100000)
        
        series.append({
            "timestamp": current_time,
            "open": round(open_price, 6),
            "high": round(high, 6),
            "low": round(low, 6),
            "close": round(current_price, 6),
            "volume": round(volume, 2),
        })
        
        current_time += interval_seconds
    
    return series


def get_mock_funding_rates(markets: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    if markets is None:
        markets = list(MOCK_PERP_MARKETS.keys())
    
    rates = {}
    for market in markets:
        if market not in MOCK_PERP_MARKETS:
            continue
        
        config = MOCK_PERP_MARKETS[market]
        base_rate = config["funding_rate"]
        
        funding_noise = _deterministic_value(market, "funding_rate", -0.00005, 0.00005)
        funding_rate = base_rate + funding_noise
        funding_apy = funding_rate * 24 * 365 * 100
        
        predicted_mult = 1 + _deterministic_value(market, "predicted", -0.2, 0.2)
        
        rates[market] = {
            "market": market,
            "funding_rate": round(funding_rate, 8),
            "funding_apy": round(funding_apy, 2),
            "next_funding_time": time.time() + 3600,
            "predicted_rate": round(funding_rate * predicted_mult, 8),
        }
    
    return rates


def get_mock_jupiter_quote(
    input_mint: str = "SOL",
    output_mint: str = "USDC",
    amount: float = 1.0
) -> Dict[str, Any]:
    input_price = MOCK_BASE_PRICES.get(input_mint, 1.0)
    output_price = MOCK_BASE_PRICES.get(output_mint, 1.0)
    
    exchange_rate = input_price / output_price
    
    pair_key = f"{input_mint}_{output_mint}"
    slippage_bps = _deterministic_value(pair_key, "slippage", 5, 50)
    effective_rate = exchange_rate * (1 - slippage_bps / 10000)
    
    output_amount = amount * effective_rate
    
    slot_hash = int(hashlib.md5(f"{pair_key}_{int(time.time() // 60)}_slot".encode()).hexdigest()[:8], 16)
    context_slot = 200000000 + int((slot_hash / 0xFFFFFFFF) * 100000000)
    
    time_taken = _deterministic_value(pair_key, "time", 0.1, 0.5)
    
    return {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "inAmount": amount,
        "outAmount": round(output_amount, 8),
        "otherAmountThreshold": round(output_amount * 0.995, 8),
        "swapMode": "ExactIn",
        "slippageBps": int(slippage_bps),
        "priceImpactPct": round(slippage_bps / 100, 4),
        "routePlan": [
            {
                "swapInfo": {
                    "ammKey": "mock_amm_key",
                    "label": "Orca",
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "inAmount": str(int(amount * 1e9)),
                    "outAmount": str(int(output_amount * 1e6)),
                    "feeAmount": str(int(amount * 1e9 * 0.003)),
                    "feeMint": input_mint,
                },
                "percent": 100,
            }
        ],
        "contextSlot": context_slot,
        "timeTaken": round(time_taken, 3),
        "source": "mock",
    }


def get_mock_balances(initial_nav: float = 10000.0) -> Dict[str, Dict[str, Any]]:
    sol_allocation = 0.1
    usdc_allocation = 0.9
    
    sol_price = MOCK_BASE_PRICES.get("SOL", 125.0)
    
    sol_value = initial_nav * sol_allocation
    usdc_value = initial_nav * usdc_allocation
    
    return {
        "SOL": {
            "token": "SOL",
            "amount": round(sol_value / sol_price, 6),
            "value_usd": round(sol_value, 2),
        },
        "USDC": {
            "token": "USDC",
            "amount": round(usdc_value, 2),
            "value_usd": round(usdc_value, 2),
        },
    }
