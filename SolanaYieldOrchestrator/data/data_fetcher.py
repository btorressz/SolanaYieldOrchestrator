from __future__ import annotations

import time
import requests
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config import Config
from utils.logging_utils import get_logger
from data.mock_data import (
    get_mock_prices,
    get_mock_perp_data,
    get_mock_funding_rates,
    get_mock_balances,
)
from infra.redis_client import RedisCache

logger = get_logger(__name__)


def _to_float(x: Any, default: float = 0.0) -> float:
    """Safe float() for Optional/unknown values (keeps pyright happy)."""
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x: Any, default: int = 0) -> int:
    """Safe int() for Optional/unknown values (keeps pyright happy)."""
    if x is None:
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


@dataclass
class PriceData:
    symbol: str
    price: float
    source: str
    timestamp: float = field(default_factory=time.time)
    change_24h: float = 0.0
    volume_24h: float = 0.0


@dataclass
class FundingData:
    market: str
    funding_rate: float
    funding_apy: float
    next_funding_time: Optional[float] = None


@dataclass
class BalanceData:
    token: str
    amount: float
    value_usd: float


@dataclass
class MarketSnapshot:
    timestamp: float
    spot_prices: Dict[str, PriceData]
    perp_prices: Dict[str, float]
    funding_rates: Dict[str, FundingData]
    balances: Dict[str, BalanceData]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": _to_float(self.timestamp),
            "timestamp_iso": datetime.fromtimestamp(_to_float(self.timestamp)).isoformat(),
            "spot_prices": {
                k: {
                    "price": _to_float(v.price),
                    "source": str(v.source),
                    "change_24h": _to_float(v.change_24h),
                }
                for k, v in self.spot_prices.items()
            },
            "perp_prices": {k: _to_float(v) for k, v in self.perp_prices.items()},
            "funding_rates": {k: {"rate": _to_float(v.funding_rate), "apy": _to_float(v.funding_apy)} for k, v in self.funding_rates.items()},
            "balances": {k: {"amount": _to_float(v.amount), "value_usd": _to_float(v.value_usd)} for k, v in self.balances.items()},
            "metadata": self.metadata,
        }


class DataFetcher:
    def __init__(self, solana_client=None, jupiter_client=None, drift_client=None, hyperliquid_client=None):
        self.solana_client = solana_client
        self.jupiter_client = jupiter_client
        self.drift_client = drift_client
        self.hyperliquid_client = hyperliquid_client

        self.coingecko_url = str(Config.COINGECKO_API_URL)
        self.kraken_url = str(Config.KRAKEN_API_URL)
        self.timeout = _to_int(getattr(Config, "API_TIMEOUT_SECONDS", 10), default=10)

        self.session = requests.Session()
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 10

        self._source_status: Dict[str, Dict[str, Any]] = {
            "jupiter": {"available": True, "last_error": None, "fail_count": 0},
            "coingecko": {"available": True, "last_error": None, "fail_count": 0},
            "kraken": {"available": True, "last_error": None, "fail_count": 0},
            "drift": {"available": True, "last_error": None, "fail_count": 0},
            "hyperliquid": {"available": True, "last_error": None, "fail_count": 0},
        }
        self._max_consecutive_failures = 3

    # ---- Source health / mode helpers ----

    def _should_use_mock(self, source: Optional[str] = None) -> bool:
        mode = str(Config.MARKET_DATA_MODE).lower()

        if mode == "mock":
            return True

        if mode == "live":
            return False

        if source is not None and source in self._source_status and not bool(self._source_status[source]["available"]):
            return True

        return False

    def _record_source_failure(self, source: str, error: str) -> None:
        self._source_status[source]["fail_count"] = _to_int(self._source_status[source].get("fail_count"), default=0) + 1
        self._source_status[source]["last_error"] = error

        if _to_int(self._source_status[source]["fail_count"]) >= _to_int(self._max_consecutive_failures):
            self._source_status[source]["available"] = False
            logger.warning(f"Source {source} marked unavailable after {self._max_consecutive_failures} consecutive failures")

    def _record_source_success(self, source: str) -> None:
        self._source_status[source]["fail_count"] = 0
        self._source_status[source]["available"] = True
        self._source_status[source]["last_error"] = None

    def _get_data_mode_info(self) -> Dict[str, Any]:
        config_mode = str(Config.MARKET_DATA_MODE).lower()

        if config_mode == "mock":
            return {"data_mode": "mock", "mock_reason": "MARKET_DATA_MODE=mock"}

        unavailable = [s for s, info in self._source_status.items() if not bool(info["available"])]

        if config_mode == "live":
            if unavailable:
                return {"data_mode": "live", "degraded": True, "unavailable_sources": unavailable}
            return {"data_mode": "live"}

        if unavailable:
            if len(unavailable) == len(self._source_status):
                return {"data_mode": "mock", "mock_reason": "All sources unavailable", "unavailable_sources": unavailable}
            return {
                "data_mode": "mixed",
                "mock_reason": f"Sources unavailable: {', '.join(unavailable)}",
                "unavailable_sources": unavailable,
            }

        return {"data_mode": "live"}

    # ---- Local in-memory cache ----

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        cached_time = _to_float(self._cache.get(f"{key}_time"), default=0.0)
        return (time.time() - cached_time) < _to_float(self._cache_ttl, default=10.0)

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._cache[f"{key}_time"] = time.time()

    # ---- Mock helpers ----

    def _get_mock_spot_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, PriceData]:
        sym_list: List[str] = list(symbols) if symbols is not None else list(Config.SUPPORTED_ASSETS)
        mock_prices = get_mock_prices(sym_list)
        prices: Dict[str, PriceData] = {}

        for symbol, data in mock_prices.items():
            prices[str(symbol)] = PriceData(
                symbol=str(symbol),
                price=_to_float(data.get("price"), default=0.0),
                source="mock",
                change_24h=_to_float(data.get("change_24h"), default=0.0),
                volume_24h=_to_float(data.get("volume_24h"), default=0.0),
            )
        return prices

    # ---- Spot price fetchers ----

    def get_coingecko_prices(self, ids: Optional[List[str]] = None, symbols: Optional[List[str]] = None) -> Dict[str, PriceData]:
        ids_list: List[str] = list(ids) if ids is not None else list(Config.ASSET_COINGECKO_IDS.values())
        symbols_list: List[str] = list(symbols) if symbols is not None else list(Config.SUPPORTED_ASSETS)

        if self._should_use_mock("coingecko"):
            logger.debug("Using mock data for CoinGecko")
            return self._get_mock_spot_prices(symbols_list)

        cache_key = f"coingecko_{','.join(ids_list)}"
        if self._is_cache_valid(cache_key):
            cached = self._cache.get(cache_key)
            return cached if isinstance(cached, dict) else {}

        try:
            params: Dict[str, Any] = {
                "ids": ",".join(ids_list),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
            }

            api_key = getattr(Config, "COINGECKO_API_KEY", None)
            if api_key:
                params["x_cg_demo_api_key"] = api_key

            response = self.session.get(f"{self.coingecko_url}/simple/price", params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json() or {}

            prices: Dict[str, PriceData] = {}
            id_to_symbol = {v: k for k, v in Config.ASSET_COINGECKO_IDS.items()}

            for coin_id, price_data in data.items():
                pdict = price_data or {}
                symbol = str(id_to_symbol.get(coin_id, str(coin_id).upper()))
                usd = _to_float(pdict.get("usd"), default=0.0)
                chg = _to_float(pdict.get("usd_24h_change"), default=0.0)
                vol = _to_float(pdict.get("usd_24h_vol"), default=0.0)

                prices[symbol] = PriceData(symbol=symbol, price=usd, source="coingecko", change_24h=chg, volume_24h=vol)

                RedisCache.set_price(f"cg:{symbol}", {"price": usd, "change_24h": chg, "source": "coingecko"})

            self._set_cache(cache_key, prices)
            self._record_source_success("coingecko")
            return prices

        except Exception as e:
            logger.error(f"CoinGecko API error: {e}")
            self._record_source_failure("coingecko", str(e))

            if str(Config.MARKET_DATA_MODE).lower() == "live":
                return {}

            return self._get_mock_spot_prices(symbols_list)

    def get_kraken_prices(self, pairs: Optional[List[str]] = None, symbols: Optional[List[str]] = None) -> Dict[str, PriceData]:
        pairs_list: List[str] = list(pairs) if pairs is not None else list(Config.ASSET_KRAKEN_PAIRS.values())
        symbols_list: List[str] = list(symbols) if symbols is not None else list(Config.ASSET_KRAKEN_PAIRS.keys())

        if self._should_use_mock("kraken"):
            logger.debug("Using mock data for Kraken")
            return self._get_mock_spot_prices(symbols_list)

        cache_key = f"kraken_{','.join(pairs_list)}"
        if self._is_cache_valid(cache_key):
            cached = self._cache.get(cache_key)
            return cached if isinstance(cached, dict) else {}

        try:
            params = {"pair": ",".join(pairs_list)}
            response = self.session.get(f"{self.kraken_url}/Ticker", params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json() or {}

            if data.get("error"):
                logger.warning(f"Kraken API error: {data['error']}")
                self._record_source_failure("kraken", str(data["error"]))
                if str(Config.MARKET_DATA_MODE).lower() != "live":
                    return self._get_mock_spot_prices(["SOL", "BTC", "ETH"])
                return {}

            prices: Dict[str, PriceData] = {}
            symbol_map = {pair: sym for sym, pair in Config.ASSET_KRAKEN_PAIRS.items()}
            symbol_map.update({"XXBTZUSD": "BTC", "XETHZUSD": "ETH"})

            for pair, ticker in (data.get("result", {}) or {}).items():
                t = ticker or {}
                symbol = str(symbol_map.get(pair, str(pair).replace("USD", "")))

                c_list = t.get("c", [0])
                o_val = t.get("o", None)
                v_list = t.get("v", [0, 0])

                last_price = _to_float(c_list[0] if isinstance(c_list, list) and c_list else 0.0, default=0.0)
                open_price = _to_float(o_val, default=last_price)

                change_24h = _to_float(((last_price - open_price) / open_price) * 100.0, default=0.0) if open_price > 0 else 0.0
                volume_24h = _to_float(v_list[1] if isinstance(v_list, list) and len(v_list) > 1 else 0.0, default=0.0)

                prices[symbol] = PriceData(
                    symbol=symbol,
                    price=last_price,
                    source="kraken",
                    change_24h=change_24h,
                    volume_24h=volume_24h,
                )

                RedisCache.set_price(f"kr:{symbol}", {"price": last_price, "change_24h": change_24h, "source": "kraken"})

            self._set_cache(cache_key, prices)
            self._record_source_success("kraken")
            return prices

        except Exception as e:
            logger.error(f"Kraken API error: {e}")
            self._record_source_failure("kraken", str(e))

            if str(Config.MARKET_DATA_MODE).lower() == "live":
                return {}

            return self._get_mock_spot_prices(symbols_list)

    def get_hyperliquid_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, PriceData]:
        symbols_list: List[str] = list(symbols) if symbols is not None else list(Config.ASSET_HL_SYMBOLS.keys())

        if self._should_use_mock("hyperliquid") or not bool(Config.HYPERLIQUID_ENABLED):
            logger.debug("Using mock data for Hyperliquid")
            allowed = [s for s in symbols_list if s in Config.ASSET_HL_SYMBOLS]
            return self._get_mock_spot_prices(allowed)

        cache_key = f"hyperliquid_{','.join(symbols_list)}"
        if self._is_cache_valid(cache_key):
            cached = self._cache.get(cache_key)
            return cached if isinstance(cached, dict) else {}

        prices: Dict[str, PriceData] = {}

        if self.hyperliquid_client:
            try:
                for symbol in symbols_list:
                    hl_symbol = Config.ASSET_HL_SYMBOLS.get(symbol)
                    if not hl_symbol:
                        continue

                    market = self.hyperliquid_client.get_market_data(hl_symbol)
                    if market:
                        mark_price = _to_float(getattr(market, "mark_price", None), default=0.0)
                        vol_24h = _to_float(getattr(market, "volume_24h", None), default=0.0)

                        prices[str(symbol)] = PriceData(
                            symbol=str(symbol),
                            price=mark_price,
                            source="hyperliquid",
                            change_24h=0.0,
                            volume_24h=vol_24h,
                        )

                        RedisCache.set_price(
                            f"hl:{symbol}",
                            {
                                "price": mark_price,
                                "source": "hyperliquid",
                                "funding_rate": _to_float(getattr(market, "funding_rate", None), default=0.0),
                                "funding_apy": _to_float(getattr(market, "funding_rate_apy", None), default=0.0),
                            },
                        )

                self._record_source_success("hyperliquid")

            except Exception as e:
                logger.error(f"Hyperliquid price fetch error: {e}")
                self._record_source_failure("hyperliquid", str(e))

                if str(Config.MARKET_DATA_MODE).lower() != "live":
                    allowed = [s for s in symbols_list if s in Config.ASSET_HL_SYMBOLS]
                    return self._get_mock_spot_prices(allowed)

        self._set_cache(cache_key, prices)
        return prices

    def get_jupiter_prices(self) -> Dict[str, PriceData]:
        if self._should_use_mock("jupiter"):
            logger.debug("Using mock data for Jupiter")
            return self._get_mock_spot_prices(["SOL", "USDC"])

        cache_key = "jupiter_prices"
        if self._is_cache_valid(cache_key):
            cached = self._cache.get(cache_key)
            return cached if isinstance(cached, dict) else {}

        prices: Dict[str, PriceData] = {}

        if self.jupiter_client:
            try:
                sol_price = self.jupiter_client.get_sol_price()
                if sol_price is not None:
                    sol_p = _to_float(sol_price, default=0.0)
                    prices["SOL"] = PriceData(symbol="SOL", price=sol_p, source="jupiter")
                    RedisCache.set_price("jup:SOL", {"price": sol_p, "source": "jupiter"})

                prices["USDC"] = PriceData(symbol="USDC", price=1.0, source="jupiter")
                self._record_source_success("jupiter")

            except Exception as e:
                logger.error(f"Jupiter price fetch error: {e}")
                self._record_source_failure("jupiter", str(e))

                if str(Config.MARKET_DATA_MODE).lower() != "live":
                    return self._get_mock_spot_prices(["SOL", "USDC"])
        else:
            if str(Config.MARKET_DATA_MODE).lower() != "live":
                return self._get_mock_spot_prices(["SOL", "USDC"])

        self._set_cache(cache_key, prices)
        return prices

    # ---- Perp + funding fetchers ----

    def _get_mock_perp_data(self) -> Tuple[Dict[str, float], Dict[str, FundingData]]:
        mock_perp = get_mock_perp_data()
        mock_funding = get_mock_funding_rates()

        perp_prices: Dict[str, float] = {}
        funding_rates: Dict[str, FundingData] = {}

        for market, data in mock_perp.items():
            perp_prices[str(market)] = _to_float((data or {}).get("mark_price"), default=0.0)

        for market, data in mock_funding.items():
            d = data or {}
            funding_rates[str(market)] = FundingData(
                market=str(market),
                funding_rate=_to_float(d.get("funding_rate"), default=0.0),
                funding_apy=_to_float(d.get("funding_apy"), default=0.0),
                next_funding_time=(_to_float(d.get("next_funding_time"), default=0.0) if d.get("next_funding_time") is not None else None),
            )

        return perp_prices, funding_rates

    def get_drift_data(self) -> Tuple[Dict[str, float], Dict[str, FundingData]]:
        if self._should_use_mock("drift"):
            logger.debug("Using mock data for Drift")
            return self._get_mock_perp_data()

        perp_prices: Dict[str, float] = {}
        funding_rates: Dict[str, FundingData] = {}

        if self.drift_client:
            try:
                markets = self.drift_client.get_perp_markets() or []

                for market in markets:
                    sym = str(getattr(market, "symbol", "") or "")
                    perp_prices[sym] = _to_float(getattr(market, "mark_price", None), default=0.0)
                    funding_rates[sym] = FundingData(
                        market=sym,
                        funding_rate=_to_float(getattr(market, "funding_rate", None), default=0.0),
                        funding_apy=_to_float(getattr(market, "funding_rate_apy", None), default=0.0),
                    )

                    RedisCache.set_perp_data(
                        sym,
                        {
                            "mark_price": perp_prices[sym],
                            "funding_rate": _to_float(getattr(market, "funding_rate", None), default=0.0),
                            "funding_apy": _to_float(getattr(market, "funding_rate_apy", None), default=0.0),
                        },
                    )

                self._record_source_success("drift")

            except Exception as e:
                logger.error(f"Drift data fetch error: {e}")
                self._record_source_failure("drift", str(e))

                if str(Config.MARKET_DATA_MODE).lower() != "live":
                    return self._get_mock_perp_data()
        else:
            if str(Config.MARKET_DATA_MODE).lower() != "live":
                return self._get_mock_perp_data()

        return perp_prices, funding_rates

    # ---- Balances ----

    def get_balances(self, initial_nav: float = 10000.0) -> Dict[str, BalanceData]:
        balances: Dict[str, BalanceData] = {}

        if bool(Config.is_simulation()):
            mock_balances = get_mock_balances(_to_float(initial_nav, default=10000.0))
            for token, data in mock_balances.items():
                d = data or {}
                balances[str(token)] = BalanceData(
                    token=str(token),
                    amount=_to_float(d.get("amount"), default=0.0),
                    value_usd=_to_float(d.get("value_usd"), default=0.0),
                )
            return balances

        if self.solana_client:
            try:
                sol_balance = _to_float(self.solana_client.get_balance(), default=0.0)
                sol_price = 100.0

                jupiter_prices = self.get_jupiter_prices()
                if "SOL" in jupiter_prices:
                    sol_price = _to_float(jupiter_prices["SOL"].price, default=100.0)

                balances["SOL"] = BalanceData(token="SOL", amount=sol_balance, value_usd=_to_float(sol_balance * sol_price, default=0.0))

            except Exception as e:
                logger.error(f"Balance fetch error: {e}")

        return balances

    # ---- Snapshot / aggregation ----

    def get_market_snapshot(self, initial_nav: float = 10000.0) -> MarketSnapshot:
        spot_prices: Dict[str, PriceData] = {}
        sources_used: List[str] = []
        mock_sources: List[str] = []

        jupiter_prices = self.get_jupiter_prices()
        if jupiter_prices:
            spot_prices.update(jupiter_prices)
            if any(p.source == "mock" for p in jupiter_prices.values()):
                mock_sources.append("jupiter")
            else:
                sources_used.append("jupiter")

        coingecko_prices = self.get_coingecko_prices()
        for symbol, price_data in coingecko_prices.items():
            if symbol not in spot_prices:
                spot_prices[symbol] = price_data
        if coingecko_prices:
            if any(p.source == "mock" for p in coingecko_prices.values()):
                mock_sources.append("coingecko")
            else:
                sources_used.append("coingecko")

        kraken_prices = self.get_kraken_prices()
        if kraken_prices:
            if any(p.source == "mock" for p in kraken_prices.values()):
                mock_sources.append("kraken")
            else:
                sources_used.append("kraken")

        if bool(Config.HYPERLIQUID_ENABLED) or str(Config.MARKET_DATA_MODE).lower() != "live":
            hl_prices = self.get_hyperliquid_prices()
            for symbol, price_data in hl_prices.items():
                if symbol not in spot_prices:
                    spot_prices[symbol] = price_data
            if hl_prices:
                if any(p.source == "mock" for p in hl_prices.values()):
                    mock_sources.append("hyperliquid")
                else:
                    sources_used.append("hyperliquid")

        perp_prices, funding_rates = self.get_drift_data()
        if perp_prices:
            if not bool(self._source_status["drift"]["available"]):
                mock_sources.append("drift")
            else:
                sources_used.append("drift")

        balances = self.get_balances(_to_float(initial_nav, default=10000.0))
        data_mode_info = self._get_data_mode_info()

        metadata: Dict[str, Any] = {
            "mode": Config.MODE,
            "market_data_mode": Config.MARKET_DATA_MODE,
            "sources": sources_used,
            "mock_sources": mock_sources,
            "kraken_prices": {k: _to_float(v.price) for k, v in kraken_prices.items()},
            **data_mode_info,
        }

        snapshot = MarketSnapshot(
            timestamp=time.time(),
            spot_prices=spot_prices,
            perp_prices=perp_prices,
            funding_rates=funding_rates,
            balances=balances,
            metadata=metadata,
        )

        RedisCache.set_snapshot(snapshot.to_dict())
        return snapshot

    def get_aggregated_price(self, symbol: str, include_pyth: bool = True) -> Optional[Dict[str, Any]]:
        sym = str(symbol)

        jupiter = self.get_jupiter_prices().get(sym)
        coingecko = self.get_coingecko_prices().get(sym)
        kraken = self.get_kraken_prices().get(sym)
        hyperliquid = self.get_hyperliquid_prices().get(sym) if sym in Config.ASSET_HL_SYMBOLS else None

        prices: List[float] = []
        sources: Dict[str, float] = {}

        if jupiter:
            jp = _to_float(jupiter.price, default=0.0)
            prices.append(jp)
            sources["jupiter"] = jp
        if coingecko:
            cg = _to_float(coingecko.price, default=0.0)
            prices.append(cg)
            sources["coingecko"] = cg
        if kraken:
            kr = _to_float(kraken.price, default=0.0)
            prices.append(kr)
            sources["kraken"] = kr
        if hyperliquid:
            hl = _to_float(hyperliquid.price, default=0.0)
            prices.append(hl)
            sources["hyperliquid"] = hl

        if not prices:
            return None

        avg_price = _to_float(sum(prices) / len(prices), default=0.0)
        max_deviation = _to_float(max((abs(p - avg_price) / avg_price * 100.0) for p in prices) if avg_price > 0 else 0.0, default=0.0)

        result: Dict[str, Any] = {
            "symbol": sym,
            "average_price": avg_price,
            "sources": sources,
            "num_sources": _to_int(len(prices), default=0),
            "max_deviation_pct": max_deviation,
            "data_mode": self._get_data_mode_info(),
        }

        if include_pyth and bool(Config.PYTH_ENABLED):
            try:
                from infra.pyth_client import get_pyth_client

                pyth = get_pyth_client()
                pyth_price = pyth.get_pyth_price(sym)

                if pyth_price is not None:
                    pyth_p = _to_float(pyth_price, default=0.0)
                    result["pyth_price"] = pyth_p
                    # FIX: ensure float(...) never receives Optional
                    result["oracle_deviation_bps"] = _to_float(pyth.get_pyth_deviation(sym, avg_price), default=0.0)
                    result["oracle_confidence_bps"] = _to_float(pyth.get_oracle_confidence_bps(sym, avg_price), default=0.0)
                    result["oracle_status"] = pyth.get_oracle_status(sym, avg_price)
            except Exception as e:
                logger.debug(f"Pyth price fetch failed for {sym}: {e}")

        return result

    # ---- Status / admin ----

    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            source: {
                "available": bool(info["available"]),
                "fail_count": _to_int(info.get("fail_count"), default=0),
                "last_error": info.get("last_error"),
            }
            for source, info in self._source_status.items()
        }

    def reset_source_status(self, source: Optional[str] = None) -> None:
        if source is not None:
            src = str(source)
            if src in self._source_status:
                self._source_status[src] = {"available": True, "last_error": None, "fail_count": 0}
        else:
            for s in list(self._source_status.keys()):
                self._source_status[s] = {"available": True, "last_error": None, "fail_count": 0}
