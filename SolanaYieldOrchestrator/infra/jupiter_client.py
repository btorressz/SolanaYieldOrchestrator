import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, cast

import requests

from config import Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
MSOL_MINT = "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"

COMMON_TOKENS: Dict[str, str] = {
    "SOL": SOL_MINT,
    "USDC": USDC_MINT,
    "mSOL": MSOL_MINT,
    "BONK": BONK_MINT,
}


def _to_float(x: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float with safe default (prevents None -> float typing issues)."""
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x: Any, default: int = 0) -> int:
    """Best-effort conversion to int with safe default."""
    if x is None:
        return int(default)
    try:
        # Handles numeric strings too.
        return int(float(x))
    except Exception:
        return int(default)


def _as_dict(x: Any) -> Dict[str, Any]:
    """Return x as dict if it is a dict, else {}. Helps pyright avoid 'str has no member get'."""
    return x if isinstance(x, dict) else {}


@dataclass
class SwapQuote:
    input_mint: str
    output_mint: str
    input_amount: int
    output_amount: int
    price: float
    price_impact_pct: float
    slippage_bps: int
    route_plan: List[Dict[str, Any]]
    raw_quote: Dict[str, Any]


@dataclass
class SwapResult:
    success: bool
    signature: Optional[str] = None
    error: Optional[str] = None
    simulated: bool = False


class JupiterClient:
    def __init__(self):
        self.base_url = Config.JUPITER_API_URL
        self.api_key = Config.JUPITER_API_KEY
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        only_direct_routes: bool = False,
    ) -> Optional[SwapQuote]:
        try:
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": slippage_bps,
                "onlyDirectRoutes": str(only_direct_routes).lower(),
            }

            response = self.session.get(f"{self.base_url}/quote", params=params, timeout=10)
            response.raise_for_status()
            data = _as_dict(response.json())

            if "error" in data:
                logger.error(f"Jupiter quote error: {data.get('error')}")
                return None

            input_amount = _to_int(data.get("inAmount", amount), amount)
            output_amount = _to_int(data.get("outAmount", 0), 0)

            price = (output_amount / input_amount) if (input_amount > 0 and output_amount > 0) else 0.0

            return SwapQuote(
                input_mint=input_mint,
                output_mint=output_mint,
                input_amount=input_amount,
                output_amount=output_amount,
                price=float(price),
                price_impact_pct=_to_float(data.get("priceImpactPct", 0.0), 0.0),
                slippage_bps=slippage_bps,
                route_plan=cast(List[Dict[str, Any]], data.get("routePlan", []) or []),
                raw_quote=data,
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Jupiter quote request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Jupiter quote error: {e}")
            return None

    def get_price(self, token_symbol: str, vs_currency: str = "USDC") -> Optional[float]:
        input_mint = COMMON_TOKENS.get(token_symbol.upper())
        output_mint = COMMON_TOKENS.get(vs_currency.upper(), USDC_MINT)

        if not input_mint:
            logger.warning(f"Unknown token symbol: {token_symbol}")
            return None

        if token_symbol.upper() == vs_currency.upper():
            return 1.0

        decimals = 9 if token_symbol.upper() in ["SOL", "mSOL"] else 6
        amount = 10**decimals

        quote = self.get_quote(input_mint, output_mint, amount)
        if quote:
            output_decimals = 6
            return float(quote.output_amount) / float(10**output_decimals)
        return None

    def get_sol_price(self) -> Optional[float]:
        return self.get_price("SOL", "USDC")

    def get_swap_transaction(
        self,
        quote: SwapQuote,
        user_public_key: str,
        wrap_unwrap_sol: bool = True,
    ) -> Optional[Dict[str, Any]]:
        try:
            payload = {
                "quoteResponse": quote.raw_quote,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": wrap_unwrap_sol,
            }

            response = self.session.post(f"{self.base_url}/swap", json=payload, timeout=10)
            response.raise_for_status()
            data = _as_dict(response.json())

            if "error" in data:
                logger.error(f"Jupiter swap error: {data.get('error')}")
                return None

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Jupiter swap request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Jupiter swap error: {e}")
            return None

    def get_token_list(self) -> List[Dict[str, Any]]:
        try:
            response = self.session.get("https://token.jup.ag/all", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Failed to get token list: {e}")
            return []

    def get_indexed_route_map(self) -> Dict[str, List[str]]:
        try:
            response = self.session.get(f"{self.base_url}/indexed-route-map", timeout=10)
            response.raise_for_status()
            data = _as_dict(response.json())
            # Keep return type stable
            return cast(Dict[str, List[str]], data)
        except Exception as e:
            logger.error(f"Failed to get route map: {e}")
            return {}


class JupiterPriceAPI:
    def __init__(self):
        self.base_url = "https://price.jup.ag/v6"
        self.session = requests.Session()

    def get_price(self, token_ids: List[str]) -> Dict[str, float]:
        try:
            params = {"ids": ",".join(token_ids)}
            response = self.session.get(f"{self.base_url}/price", params=params, timeout=10)
            response.raise_for_status()
            data = _as_dict(response.json())

            prices: Dict[str, float] = {}
            for token_id, info_any in _as_dict(data.get("data", {})).items():
                info = _as_dict(info_any)
                # FIX: ensures float even if API returns None
                prices[str(token_id)] = _to_float(info.get("price", 0.0), 0.0)
            return prices

        except Exception as e:
            logger.error(f"Jupiter price API error: {e}")
            return {}

    def get_sol_usdc_price(self) -> float:
        prices = self.get_price([SOL_MINT])
        return _to_float(prices.get(SOL_MINT, 0.0), 0.0)


@dataclass
class RouteHop:
    pool_address: str
    pool_type: str
    input_mint: str
    output_mint: str
    fee_bps: float
    liquidity: float
    expected_output: int


@dataclass
class RouteInfo:
    market: str
    input_mint: str
    output_mint: str
    input_amount: int
    output_amount: int
    price: float
    price_impact_pct: float
    total_fee_bps: float
    hops: List[RouteHop]
    route_labels: List[str]
    is_direct: bool
    estimated_slippage_bps: float
    mev_risk_level: str = "low"  # "low", "medium", "high"
    safe_mode_allowed: bool = True


class JupiterHealthMetrics:
    def __init__(self, max_samples: int = 100):
        self._latencies: List[float] = []
        self._failures: int = 0
        self._total: int = 0
        self._last_error: Optional[str] = None
        self._last_error_ts: Optional[float] = None
        self._max_samples = max_samples
        self._lock = __import__("threading").Lock()

    def record_request(self, latency_ms: float, success: bool, error: Optional[str] = None):
        with self._lock:
            self._total += 1
            if success:
                self._latencies.append(latency_ms)
                if len(self._latencies) > self._max_samples:
                    self._latencies = self._latencies[-self._max_samples :]
            else:
                self._failures += 1
                self._last_error = error
                self._last_error_ts = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            if not self._latencies:
                return {
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "failure_rate": 0.0 if self._total == 0 else float(self._failures) / float(self._total),
                    "total_requests": self._total,
                    "last_error": self._last_error,
                    "last_error_timestamp": self._last_error_ts,
                }

            sorted_lat = sorted(self._latencies)
            p95_idx = int(len(sorted_lat) * 0.95)

            return {
                "avg_latency_ms": round(sum(self._latencies) / len(self._latencies), 2),
                "p95_latency_ms": round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 2),
                "failure_rate": round(self._failures / self._total if self._total > 0 else 0.0, 4),
                "total_requests": self._total,
                "last_error": self._last_error,
                "last_error_timestamp": self._last_error_ts,
            }

    def get_status(self) -> str:
        metrics = self.get_metrics()
        failure_rate = _to_float(metrics.get("failure_rate", 0.0), 0.0)
        p95 = _to_float(metrics.get("p95_latency_ms", 0.0), 0.0)

        if failure_rate > 0.5:
            return "failing"
        if failure_rate > 0.1 or p95 > 2000:
            return "degraded"
        return "good"


@dataclass
class RouteDepthInfo:
    route_id: str
    route_description: str
    price: float
    slippage_bps: float
    depth_to_1pct_slippage: float
    fees_bps: float
    hops: int
    dexes: List[str]


@dataclass
class SlippageCurvePoint:
    size: float
    effective_price: float
    slippage_bps: float
    price_impact_pct: float


class EnhancedJupiterClient(JupiterClient):
    def __init__(self):
        super().__init__()
        self.sanity_threshold_bps = 100.0
        self.health_metrics = JupiterHealthMetrics()

    def get_route_info(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
    ) -> Optional[RouteInfo]:
        quote = self.get_quote(input_mint, output_mint, amount, slippage_bps)
        if not quote:
            return None
        return self._parse_route_info(quote)

    def _parse_route_info(self, quote: SwapQuote) -> RouteInfo:
        hops: List[RouteHop] = []
        route_labels: List[str] = []
        total_fee_bps = 0.0

        for plan_step_any in quote.route_plan:
            plan_step = _as_dict(plan_step_any)
            swap_info = _as_dict(plan_step.get("swapInfo", {}))

            pool_type = str(swap_info.get("label", "unknown"))
            route_labels.append(pool_type)

            in_amt = _to_float(swap_info.get("inAmount", 1.0), 1.0)
            fee_amt = _to_float(swap_info.get("feeAmount", 0.0), 0.0)
            fee_bps = (fee_amt / max(1.0, in_amt)) * 10000.0
            total_fee_bps += fee_bps

            hop = RouteHop(
                pool_address=str(swap_info.get("ammKey", "")),
                pool_type=pool_type,
                input_mint=str(swap_info.get("inputMint", "")),
                output_mint=str(swap_info.get("outputMint", "")),
                fee_bps=float(fee_bps),
                liquidity=0.0,
                expected_output=_to_int(swap_info.get("outAmount", 0), 0),
            )
            hops.append(hop)

        input_symbol = self._get_symbol_from_mint(quote.input_mint)
        output_symbol = self._get_symbol_from_mint(quote.output_mint)
        market = f"{input_symbol}/{output_symbol}"

        mev_risk_level, safe_mode_allowed = self._assess_mev_risk(route_labels, hops, float(quote.price_impact_pct))

        return RouteInfo(
            market=market,
            input_mint=quote.input_mint,
            output_mint=quote.output_mint,
            input_amount=quote.input_amount,
            output_amount=quote.output_amount,
            price=float(quote.price),
            price_impact_pct=float(quote.price_impact_pct),
            total_fee_bps=float(total_fee_bps),
            hops=hops,
            route_labels=route_labels,
            is_direct=len(hops) <= 1,
            estimated_slippage_bps=float(quote.slippage_bps),
            mev_risk_level=str(mev_risk_level),
            safe_mode_allowed=bool(safe_mode_allowed),
        )

    def _assess_mev_risk(self, route_labels: List[str], hops: List[RouteHop], price_impact_pct: float) -> Tuple[str, bool]:
        high_mev_pools = {"Raydium CLMM", "Orca Whirlpool", "Meteora DLMM", "Lifinity"}
        medium_mev_pools = {"Raydium", "Orca", "Serum", "OpenBook"}
        safe_pools = {"Jupiter Limit Order", "Phoenix", "Raydium CP"}

        has_high_mev = any(label in high_mev_pools for label in route_labels)
        has_medium_mev = any(label in medium_mev_pools for label in route_labels)
        has_only_safe = all(label in safe_pools for label in route_labels) if route_labels else False

        if len(hops) > 3:
            return "high", False
        if has_high_mev or price_impact_pct > 1.0:
            return "high", False
        if has_medium_mev or price_impact_pct > 0.3 or len(hops) > 2:
            safe_mode_allowed = len(hops) <= 2 and price_impact_pct <= 0.5
            return "medium", safe_mode_allowed
        if has_only_safe:
            return "low", True
        return "low", True

    def _get_symbol_from_mint(self, mint: str) -> str:
        for symbol, mint_addr in COMMON_TOKENS.items():
            if mint_addr == mint:
                return symbol
        return mint[:8]

    def compare_cross_venue_prices(self, prices: Dict[str, Dict[str, float]], sanity_threshold_bps: Optional[float] = None) -> Dict[str, Any]:
        threshold = float(sanity_threshold_bps) if sanity_threshold_bps is not None else float(self.sanity_threshold_bps)
        results: Dict[str, Any] = {}

        for symbol in ["SOL", "BTC", "ETH"]:
            venue_prices: Dict[str, float] = {}

            if "jupiter" in prices and symbol in prices["jupiter"]:
                venue_prices["jupiter"] = _to_float(prices["jupiter"][symbol], 0.0)
            if "coingecko" in prices and symbol in prices["coingecko"]:
                venue_prices["coingecko"] = _to_float(prices["coingecko"][symbol], 0.0)
            if "kraken" in prices and symbol in prices["kraken"]:
                venue_prices["kraken"] = _to_float(prices["kraken"][symbol], 0.0)

            if len(venue_prices) < 2:
                results[symbol] = {
                    "prices": venue_prices,
                    "max_deviation_bps": 0.0,
                    "sanity_check": "insufficient_data",
                    "is_safe": True,
                }
                continue

            price_values = list(venue_prices.values())
            mid_price = sum(price_values) / len(price_values)

            max_deviation = 0.0
            if mid_price > 0:
                max_deviation = max(abs(p - mid_price) / mid_price * 10000.0 for p in price_values)

            is_safe = max_deviation <= threshold

            results[symbol] = {
                "prices": venue_prices,
                "mid_price": float(mid_price),
                "max_deviation_bps": float(max_deviation),
                "sanity_check": "pass" if is_safe else "fail",
                "is_safe": bool(is_safe),
                "warning": None if is_safe else f"Price deviation {max_deviation:.1f}bps exceeds threshold {threshold}bps",
            }

        overall_safe = all(bool(r.get("is_safe", True)) for r in results.values())

        return {
            "by_symbol": results,
            "overall_sanity": "pass" if overall_safe else "fail",
            "threshold_bps": threshold,
            "timestamp": time.time(),
        }

    def validate_trade_sanity(
        self,
        market: str,
        size_usd: float,
        prices: Dict[str, Dict[str, float]],
        max_size_usd: float = 50000,
    ) -> Dict[str, Any]:
        warnings: List[Dict[str, Any]] = []
        is_blocked = False

        sanity = self.compare_cross_venue_prices(prices)
        if sanity["overall_sanity"] == "fail":
            warnings.append(
                {
                    "type": "price_deviation",
                    "severity": "high",
                    "message": "Cross-venue price deviation exceeds threshold",
                }
            )
            is_blocked = True

        if size_usd > max_size_usd:
            warnings.append(
                {
                    "type": "size_limit",
                    "severity": "medium",
                    "message": f"Trade size ${size_usd:,.0f} exceeds recommended max ${max_size_usd:,.0f}",
                }
            )

        return {
            "market": market,
            "size_usd": size_usd,
            "sanity_check": sanity,
            "warnings": warnings,
            "is_blocked": is_blocked,
            "can_proceed": not is_blocked,
        }

    def to_route_info_dict(self, route_info: RouteInfo) -> Dict[str, Any]:
        return {
            "market": route_info.market,
            "input_mint": route_info.input_mint,
            "output_mint": route_info.output_mint,
            "input_amount": route_info.input_amount,
            "output_amount": route_info.output_amount,
            "price": route_info.price,
            "price_impact_pct": route_info.price_impact_pct,
            "total_fee_bps": route_info.total_fee_bps,
            "hops": [
                {
                    "pool_address": hop.pool_address,
                    "pool_type": hop.pool_type,
                    "input_mint": hop.input_mint,
                    "output_mint": hop.output_mint,
                    "fee_bps": hop.fee_bps,
                    "expected_output": hop.expected_output,
                }
                for hop in route_info.hops
            ],
            "route_labels": route_info.route_labels,
            "is_direct": route_info.is_direct,
            "estimated_slippage_bps": route_info.estimated_slippage_bps,
            "mev_risk_level": route_info.mev_risk_level,
            "safe_mode_allowed": route_info.safe_mode_allowed,
        }

    def get_quote_with_timing(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
    ) -> Optional[SwapQuote]:
        start = time.time()
        try:
            quote = self.get_quote(input_mint, output_mint, amount, slippage_bps)
            latency_ms = (time.time() - start) * 1000.0
            self.health_metrics.record_request(latency_ms, quote is not None)
            return quote
        except Exception as e:
            latency_ms = (time.time() - start) * 1000.0
            self.health_metrics.record_request(latency_ms, False, str(e))
            return None

    def get_route_depth_info(
        self,
        input_mint: str,
        output_mint: str,
        base_amount: int,
        slippage_bps: int = 50,
    ) -> Optional[RouteDepthInfo]:
        quote = self.get_quote_with_timing(input_mint, output_mint, base_amount, slippage_bps)
        if not quote:
            return None

        route_labels: List[str] = []
        for plan_step_any in quote.route_plan:
            plan_step = _as_dict(plan_step_any)
            swap_info = _as_dict(plan_step.get("swapInfo", {}))
            route_labels.append(str(swap_info.get("label", "unknown")))

        depth_estimate = self._estimate_depth_to_slippage(input_mint, output_mint, base_amount, target_slippage_bps=100.0)

        total_fee_bps = 0.0
        for plan_step_any in quote.route_plan:
            plan_step = _as_dict(plan_step_any)
            swap_info = _as_dict(plan_step.get("swapInfo", {}))
            in_amt = _to_float(swap_info.get("inAmount", 1.0), 1.0)
            fee_amt = _to_float(swap_info.get("feeAmount", 0.0), 0.0)
            if in_amt > 0:
                total_fee_bps += (fee_amt / in_amt) * 10000.0

        return RouteDepthInfo(
            route_id=f"{input_mint[:8]}_{output_mint[:8]}_{base_amount}",
            route_description=" -> ".join(route_labels) if route_labels else "Direct",
            price=float(quote.price),
            slippage_bps=float(quote.price_impact_pct) * 100.0,
            depth_to_1pct_slippage=float(depth_estimate),
            fees_bps=float(total_fee_bps),
            hops=int(len(quote.route_plan)),
            dexes=route_labels,
        )

    def _estimate_depth_to_slippage(
        self,
        input_mint: str,
        output_mint: str,
        base_amount: int,
        target_slippage_bps: float = 100.0,
    ) -> float:
        test_sizes = [base_amount * m for m in [1, 2, 5, 10, 20]]

        for test_size in test_sizes:
            quote = self.get_quote(input_mint, output_mint, int(test_size), 200)
            if quote and float(quote.price_impact_pct) * 100.0 >= target_slippage_bps:
                return float(test_size)

        return float(test_sizes[-1])

    def compare_routes(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        profile: str = "best_price",
    ) -> List[Dict[str, Any]]:
        routes: List[Dict[str, Any]] = []

        for slippage in [25, 50, 100, 200]:
            quote = self.get_quote_with_timing(input_mint, output_mint, amount, slippage)
            if quote:
                # FIX: avoid calling .get on a str by keeping swap_info as dict
                labels: List[str] = []
                for step_any in quote.route_plan:
                    step = _as_dict(step_any)
                    swap_info = _as_dict(step.get("swapInfo", {}))
                    labels.append(str(swap_info.get("label", "?")))

                routes.append(
                    {
                        "slippage_setting": slippage,
                        "price": float(quote.price),
                        "output_amount": int(quote.output_amount),
                        "price_impact_pct": float(quote.price_impact_pct),
                        "price_impact_bps": float(quote.price_impact_pct) * 100.0,
                        "route": " -> ".join(labels),
                        "hops": int(len(quote.route_plan)),
                        "is_direct": len(quote.route_plan) <= 1,
                    }
                )

        if profile == "best_price":
            routes.sort(key=lambda r: -int(r["output_amount"]))
        elif profile == "best_depth":
            routes.sort(key=lambda r: float(r["price_impact_bps"]))
        elif profile == "most_stable":
            routes.sort(key=lambda r: (int(r["hops"]), float(r["price_impact_bps"])))

        return routes

    def get_slippage_curve(
        self,
        input_mint: str,
        output_mint: str,
        min_size: int,
        max_size: int,
        steps: int = 10,
    ) -> List[SlippageCurvePoint]:
        if steps < 2:
            steps = 2

        step_size = (max_size - min_size) // (steps - 1)
        sizes = [min_size + i * step_size for i in range(steps)]

        base_quote = self.get_quote(input_mint, output_mint, min_size)
        if not base_quote:
            return []

        base_price = float(base_quote.price)

        curve: List[SlippageCurvePoint] = []
        for size in sizes:
            quote = self.get_quote(input_mint, output_mint, size, 200)
            if quote:
                slippage_bps = 0.0
                if base_price > 0:
                    slippage_bps = ((base_price - float(quote.price)) / base_price) * 10000.0

                curve.append(
                    SlippageCurvePoint(
                        size=float(size),
                        effective_price=float(quote.price),
                        slippage_bps=round(slippage_bps, 2),
                        price_impact_pct=float(quote.price_impact_pct),
                    )
                )

        return curve

    def get_impact_analysis(
        self,
        input_mint: str,
        output_mint: str,
        min_size: int,
        max_size: int,
        steps: int = 10,
    ) -> Dict[str, Any]:
        curve = self.get_slippage_curve(input_mint, output_mint, min_size, max_size, steps)
        if not curve:
            return {"success": False, "error": "Could not generate slippage curve"}

        slippages = [p.slippage_bps for p in curve]

        max_size_100bps: float = float(max_size)
        for point in curve:
            if point.slippage_bps >= 100:
                max_size_100bps = float(point.size)
                break

        return {
            "success": True,
            "input_mint": input_mint,
            "output_mint": output_mint,
            "curve": [
                {
                    "size": p.size,
                    "effective_price": p.effective_price,
                    "slippage_bps": p.slippage_bps,
                    "price_impact_pct": p.price_impact_pct,
                }
                for p in curve
            ],
            "max_size_for_100bps": max_size_100bps,
            "min_slippage_bps": min(slippages) if slippages else 0.0,
            "max_slippage_bps": max(slippages) if slippages else 0.0,
        }

    def get_health(self) -> Dict[str, Any]:
        metrics = self.health_metrics.get_metrics()
        status = self.health_metrics.get_status()
        return {"status": status, **metrics}

    def get_price_impact_curve(
        self,
        base_symbol: str,
        quote_symbol: str,
        side: str,
        max_size: float,
        points: int = 10,
    ) -> Dict[str, Any]:
        try:
            from infra.redis_client import RedisCache, is_redis_available

            if is_redis_available():
                cached = RedisCache.get_impact_curve(base_symbol, quote_symbol, side, max_size, points)
                if cached:
                    return cached
        except ImportError:
            pass

        DECIMALS_MAP: Dict[str, int] = {
            "SOL": 9,
            "mSOL": 9,
            "MSOL": 9,
            "jitoSOL": 9,
            "JITOSOL": 9,
            "USDC": 6,
            "USDT": 6,
            "BTC": 8,
            "ETH": 8,
            "RAY": 6,
            "SRM": 6,
        }

        base_mint = COMMON_TOKENS.get(base_symbol.upper())
        quote_mint = COMMON_TOKENS.get(quote_symbol.upper(), USDC_MINT)

        if not base_mint:
            return self._generate_mock_impact_curve writer(base_symbol, quote_symbol, side, max_size, points)  # type: ignore[name-defined]

        base_decimals = DECIMALS_MAP.get(base_symbol.upper(), 6)
        quote_decimals = DECIMALS_MAP.get(quote_symbol.upper(), 6)

        if side.lower() == "sell":
            input_mint = base_mint
            output_mint = quote_mint
            decimals = base_decimals
        else:
            input_mint = quote_mint
            output_mint = base_mint
            decimals = quote_decimals

        curve_points: List[Dict[str, Any]] = []
        sizes: List[float] = []
        step = float(max_size) / max(points - 1, 1)

        for i in range(points):
            size = step * (i + 1) if i > 0 else step * 0.1
            sizes.append(size)

        base_price: Optional[float] = None
        for size in sizes:
            amount = int(size * float(10**decimals))
            quote = self.get_quote_with_timing(input_mint, output_mint, amount)

            if quote:
                if base_price is None:
                    base_price = float(quote.price)

                base_price_f = base_price if base_price is not None else 0.0  # FIX: never pass None to float math
                slippage_bps = 0.0
                if base_price_f > 0:
                    price_diff = abs(float(quote.price) - base_price_f) / base_price_f
                    slippage_bps = price_diff * 10000.0

                fee_estimate = size * float(quote.price) * 0.0005

                curve_points.append(
                    {
                        "size": round(size, 4),
                        "price": round(float(quote.price), 6),
                        "slippage_bps": round(slippage_bps, 2),
                        "fees": round(fee_estimate, 4),
                    }
                )
            else:
                break

        if len(curve_points) < 2:
            return self._generate_mock_impact_curve(base_symbol, quote_symbol, side, max_size, points)

        result = {
            "base_symbol": base_symbol,
            "quote_symbol": quote_symbol,
            "side": side,
            "points": curve_points,
            "max_size": max_size,
            "source": "live",
        }

        try:
            from infra.redis_client import RedisCache, is_redis_available

            if is_redis_available():
                RedisCache.set_impact_curve(base_symbol, quote_symbol, side, max_size, result, points)
        except ImportError:
            pass

        return result

    def _generate_mock_impact_curve(
        self,
        base_symbol: str,
        quote_symbol: str,
        side: str,
        max_size: float,
        points: int = 10,
    ) -> Dict[str, Any]:
        import hashlib

        hash_input = f"{base_symbol}_{quote_symbol}_{side}_{int(time.time() // 300)}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)

        base_prices = {
            "SOL": 100.0 + (hash_val % 50),
            "BTC": 43000.0 + (hash_val % 5000),
            "ETH": 2200.0 + (hash_val % 500),
            "mSOL": 105.0 + (hash_val % 50),
        }

        base_price = float(base_prices.get(base_symbol.upper(), 100.0))

        curve_points: List[Dict[str, Any]] = []
        step = float(max_size) / max(points - 1, 1)

        for i in range(points):
            size = step * (i + 1) if i > 0 else step * 0.1

            size_factor = size / float(max_size) if max_size != 0 else 0.0
            slippage_bps = 2.0 + (size_factor**1.5) * 100.0 + float(hash_val % 10)

            effective_price = (
                base_price * (1 + slippage_bps / 10000.0)
                if side == "buy"
                else base_price * (1 - slippage_bps / 10000.0)
            )

            fees = size * base_price * 0.0005

            curve_points.append(
                {
                    "size": round(size, 4),
                    "price": round(float(effective_price), 6),
                    "slippage_bps": round(float(slippage_bps), 2),
                    "fees": round(float(fees), 4),
                }
            )

        return {
            "base_symbol": base_symbol,
            "quote_symbol": quote_symbol,
            "side": side,
            "points": curve_points,
            "max_size": max_size,
            "source": "mock",
        }

    def get_routes_comparison(
        self,
        base_symbol: str,
        quote_symbol: str,
        side: str,
        size: float,
        limit: int = 5,
    ) -> Dict[str, Any]:
        input_mint = COMMON_TOKENS.get(base_symbol.upper())
        output_mint = COMMON_TOKENS.get(quote_symbol.upper(), USDC_MINT)

        if not input_mint:
            return self._generate_mock_routes(base_symbol, quote_symbol, side, size, limit)

        if side.lower() == "sell":
            input_mint, output_mint = output_mint, input_mint

        decimals = 9 if base_symbol.upper() in ["SOL", "mSOL"] else 6
        amount = int(size * float(10**decimals))

        routes = self.compare_routes(input_mint, output_mint, amount, "best_price")

        if not routes:
            return self._generate_mock_routes(base_symbol, quote_symbol, side, size, limit)

        enhanced_routes: List[Dict[str, Any]] = []
        for route in routes[:limit]:
            impact_bps = _to_float(route.get("price_impact_bps", 1.0), 1.0)
            depth_estimate = size * (100.0 / max(impact_bps, 0.1))

            enhanced_routes.append(
                {
                    "route_id": f"{base_symbol}_{quote_symbol}_{route.get('slippage_setting', 50)}",
                    "hops": route.get("hops", 1),
                    "route_path": route.get("route", "Direct"),
                    "price": route.get("price", 0.0),
                    "est_slippage_bps": impact_bps,
                    "est_depth_to_1pct": round(depth_estimate, 2),
                    "total_fee_bps": round(impact_bps * 0.5 + 5.0, 2),
                    "is_direct": bool(route.get("is_direct", False)),
                }
            )

        return {
            "base_symbol": base_symbol,
            "quote_symbol": quote_symbol,
            "side": side,
            "size": size,
            "routes": enhanced_routes,
            "source": "live",
        }

    def _generate_mock_routes(
        self,
        base_symbol: str,
        quote_symbol: str,
        side: str,
        size: float,
        limit: int = 5,
    ) -> Dict[str, Any]:
        import hashlib

        hash_input = f"{base_symbol}_{quote_symbol}_{int(time.time() // 300)}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)

        base_prices = {"SOL": 100.0, "BTC": 43000.0, "ETH": 2200.0}
        base_price = float(base_prices.get(base_symbol.upper(), 100.0))

        dexes = ["Raydium", "Orca", "Phoenix", "Lifinity", "Meteora"]

        routes: List[Dict[str, Any]] = []
        for i in range(min(limit, 5)):
            slippage = float(5 + i * 3 + (hash_val % 10))
            price = base_price * (1 + (slippage / 10000.0) if side == "buy" else 1 - (slippage / 10000.0))

            routes.append(
                {
                    "route_id": f"{base_symbol}_{quote_symbol}_mock_{i}",
                    "hops": 1 + (i % 2),
                    "route_path": dexes[i] if i == 0 else f"{dexes[i]} -> {dexes[(i + 1) % len(dexes)]}",
                    "price": round(float(price), 6),
                    "est_slippage_bps": round(float(slippage), 2),
                    "est_depth_to_1pct": round(size * (100.0 / max(float(slippage), 0.1)), 2),
                    "total_fee_bps": round(3.0 + i * 2.0, 2),
                    "is_direct": i == 0,
                }
            )

        return {
            "base_symbol": base_symbol,
            "quote_symbol": quote_symbol,
            "side": side,
            "size": size,
            "routes": routes,
            "source": "mock",
        }

