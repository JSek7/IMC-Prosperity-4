from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, Optional, Tuple
import json
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
        debug_info: Dict[str, Any],
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    debug_info,
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = max(0, (self.max_log_length - base_length) // 3)

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    debug_info,
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        return {symbol: [od.buy_orders, od.sell_orders] for symbol, od in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        if max_length <= 3:
            return value[:max_length]

        lo, hi = 0, len(value)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid] + "..."
            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

class Trader:
    LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    PARAMS = {
    "INTARIAN_PEPPER_ROOT": {
        "take_edge": 0.0,
        "typical_spread": 8.0,
        "clip": 20,
        "inv_skew_long": 0.03,
        "inv_skew_short": 0.18,
        "hard_long": 75,
        "hard_short": 20,
        "wide_spread": 8,
        "target_scale": 12,
        "fast_alpha": 0.20,
        "slow_alpha": 0.05,
        "trend_mult": 0.2,
    },
    "ASH_COATED_OSMIUM": {
        "take_edge": 1.5,
        "typical_spread": 8.0,
        "clip": 10,
        "vol_mult": 1.2,
        "inv_skew": 0.25,
        "imbalance_mult": 0.5,
        "hard_pos": 50,
        "fast_alpha": 0.24,
        "slow_alpha": 0.07,
        "trend_mult": 0.50,
    },
}

# Helper functions for data manipulation and strategy logic

    def load_data(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {
                "ema_fast": {},
                "ema_slow": {},
                "price_hist": {},
                "osmium_fair": {},
                "root_centre": {},
                "root_regime": "neutral",
            }

        try:
            data = json.loads(trader_data)
        except Exception:
            data = {}

        if "ema_fast" not in data:
            data["ema_fast"] = {}
        if "ema_slow" not in data:
            data["ema_slow"] = {}
        if "price_hist" not in data:
            data["price_hist"] = {}
        if "osmium_fair" not in data:
            data["osmium_fair"] = {}
        if "root_centre" not in data:
            data["root_centre"] = {}
        if "root_regime" not in data:
            data["root_regime"] = "neutral"

        return data
    
    def save_data(self, data: Dict[str, Any]) -> str:
        return json.dumps(data, separators=(",", ":"))

    def get_position(self, state: TradingState, symbol: str) -> int:
        return state.position.get(symbol, 0)

    def best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def sorted_bids(self, order_depth: OrderDepth) -> List[Tuple[int, int]]:
        return sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)

    def sorted_asks(self, order_depth: OrderDepth) -> List[Tuple[int, int]]:
        return sorted(order_depth.sell_orders.items(), key=lambda x: x[0])

    def mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid) 
        if best_ask is not None:
            return float(best_ask)
        return None

    def microprice(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return self.mid_price(order_depth)

        bid_size = order_depth.buy_orders.get(best_bid, 0)
        ask_size = -order_depth.sell_orders.get(best_ask, 0)

        if bid_size <= 0 and ask_size <= 0:
            return (best_bid + best_ask) / 2.0
        if bid_size <= 0:
            return float(best_ask)
        if ask_size <= 0:
            return float(best_bid)

        return (best_bid * ask_size + best_ask * bid_size) / float(bid_size + ask_size)

    def update_ema(self, prev: Optional[float], new_value: float, alpha: float) -> float:
        if prev is None:
            return new_value
        return alpha * new_value + (1.0 - alpha) * prev

    def remaining_buy_capacity(self, limit: int, pos: int, buy_used: int, sell_used: int) -> int:
        return max(0, limit - (pos + buy_used - sell_used))

    def remaining_sell_capacity(self, limit: int, pos: int, buy_used: int, sell_used: int) -> int:
        return max(0, limit + (pos + buy_used - sell_used))

    def place_buy(
        self,
        orders: List[Order],
        symbol: str,
        price: int,
        qty: int,
        limit: int,
        pos: int,
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:
        qty = min(qty, self.remaining_buy_capacity(limit, pos, buy_used, sell_used))
        if qty > 0:
            orders.append(Order(symbol, int(price), int(qty)))
            buy_used += qty
        return buy_used, sell_used

    def place_sell(
        self,
        orders: List[Order],
        symbol: str,
        price: int,
        qty: int,
        limit: int,
        pos: int,
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:
        qty = min(qty, self.remaining_sell_capacity(limit, pos, buy_used, sell_used))
        if qty > 0:
            orders.append(Order(symbol, int(price), -int(qty)))
            sell_used += qty
        return buy_used, sell_used
    
    def realized_vol(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.0

        diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        if len(diffs) < 2:
            return abs(diffs[0]) if diffs else 0.0

        mean_diff = sum(diffs) / len(diffs)
        var = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
        return math.sqrt(var)
    
    def book_imbalance(self, order_depth: OrderDepth, levels: int = 3) -> float:
        bids = self.sorted_bids(order_depth)[:levels]
        asks = self.sorted_asks(order_depth)[:levels]

        bid_vol = sum(qty for _, qty in bids)
        ask_vol = sum(-qty for _, qty in asks)

        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0

        return (bid_vol - ask_vol) / total
    

    def get_recent_trades(self, state: TradingState, symbol: str) -> Tuple[List[Trade], List[Trade]]:
        own = state.own_trades.get(symbol, [])
        market = state.market_trades.get(symbol, [])
        return own, market


#market make root only when there is limited liquidity, otherwise take when favorable and market make on both sides with skew based on position and fair/trend

    def root_centre(self, order_depth: OrderDepth, data: Dict[str, Any]) -> Optional[float]:
        params = self.PARAMS["INTARIAN_PEPPER_ROOT"]

        best_bid, best_ask = self.best_bid_ask(order_depth)
        micro = self.microprice(order_depth)
        mid = self.mid_price(order_depth)
        imbalance = self.book_imbalance(order_depth, levels=3)

        prev_centre = data.get("root_centre", {}).get("last_centre")
        if mid is not None and prev_centre is not None:
            if abs(prev_centre - mid) > 100:
                prev_centre = mid

        ema_fast = data["ema_fast"].get("INTARIAN_PEPPER_ROOT")
        ema_slow = data["ema_slow"].get("INTARIAN_PEPPER_ROOT")

        trend = 0.0
        if ema_fast is not None and ema_slow is not None:
            trend = ema_fast - ema_slow

        # Full or reasonably informative book
        if micro is not None and mid is not None:
            if prev_centre is not None:
                anchor = 0.8 * prev_centre + 0.2 * mid
            else:
                anchor = mid

            centre = (
                0.35 * micro
                + 0.35 * mid
                + 0.3 * anchor
                + 0.2 * trend
                #+ 0.05 * imbalance
            )
            return centre

        # One-sided or sparse book fallback
        if prev_centre is not None:
            if best_bid is not None and best_ask is None:
                return 0.85 * prev_centre + 0.15 * float(best_bid) + params["trend_mult"] * trend
            if best_ask is not None and best_bid is None:
                return 0.85 * prev_centre + 0.15 * float(best_ask) + params["trend_mult"] * trend
            return prev_centre + params["trend_mult"] * trend

        # Final fallback if no previous centre exists
        if best_bid is not None:
            return float(best_bid) + params["trend_mult"] * trend
        if best_ask is not None:
            return float(best_ask) + params["trend_mult"] * trend

        return None

    def make_root(
        self,
        order_depth: OrderDepth,
        data: Dict[str, Any],
        pos: int,
        orders: List[Order],
        buy_used: int,
        sell_used: int,
        debug_info: Dict[str, Any],
    ) -> Tuple[int, int]:

        symbol = "INTARIAN_PEPPER_ROOT"
        params = self.PARAMS[symbol]
        limit = self.LIMITS[symbol]

        ema_fast = data["ema_fast"].get(symbol)
        ema_slow = data["ema_slow"].get(symbol)
        trend = 0.0
        if ema_fast is not None and ema_slow is not None:
            trend = ema_fast - ema_slow

        signal, regime = self.root_signal_and_regime(data)

        centre = self.root_centre(order_depth, data)
        best_bid, best_ask = self.best_bid_ask(order_depth)

        if centre is None:
            debug_info[symbol] = {
                "our_bid": None,
                "our_ask": None,
                "bid_size": 0,
                "ask_size": 0,
                "position": pos + buy_used - sell_used,
                "regime": regime,
                "signal": signal,
                "centre": None,
            }
            return buy_used, sell_used

        current_pos = pos + buy_used - sell_used

        if current_pos >= 0:
            pos_skew = params["inv_skew_long"] * current_pos
        else:
            pos_skew = params["inv_skew_short"] * current_pos

        spread = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid

        if best_bid is not None and best_ask is not None and spread is not None and spread <= params["wide_spread"]:
            bid_quote = best_bid + 1
            ask_quote = best_ask - 1

            if bid_quote >= ask_quote:
                bid_quote = best_bid
                ask_quote = best_ask
        else:
            half_spread = params["typical_spread"] // 2

            if trend > 0:
                bid_quote = math.floor(centre - max(1.0, half_spread - 1))
                ask_quote = math.ceil(centre + half_spread + 2)
            elif trend < 0:
                bid_quote = math.floor(centre - half_spread - 2)
                ask_quote = math.ceil(centre + max(1.0, half_spread - 1))
            else:
                bid_quote = math.floor(centre - half_spread)
                ask_quote = math.ceil(centre + half_spread)

            if best_bid is None and best_ask is not None:
                bid_quote -= 2
            if best_ask is None and best_bid is not None:
                ask_quote += 2

            if bid_quote >= ask_quote:
                c = int(round(centre))
                bid_quote = c - 3
                ask_quote = c + 3

        bid_quote = math.floor(bid_quote - pos_skew)
        ask_quote = math.ceil(ask_quote - pos_skew)

        if bid_quote >= ask_quote:
            c = int(round(centre))
            bid_quote = c - 1
            ask_quote = c + 1

        buy_cap = self.remaining_buy_capacity(limit, pos, buy_used, sell_used)
        sell_cap = self.remaining_sell_capacity(limit, pos, buy_used, sell_used)

        passive_clip = max(1, params["clip"] // 2)

        if regime == "long":
            if signal > 3.0:
                target_pos = 70
            elif signal > 2.0:
                target_pos = 50
            elif signal > 1.2:
                target_pos = 25
            else:
                target_pos = 10
        elif regime == "short":
            if signal < -3.5:
                target_pos = -20
            elif signal < -2.5:
                target_pos = -12
            else:
                target_pos = -6
        else:
            target_pos = 0

        buy_qty = passive_clip
        sell_qty = passive_clip

        if regime == "long":
            if current_pos < target_pos - 5:
                sell_qty = 0
                buy_qty = min(buy_cap, passive_clip + 4)
                ask_quote += 4
            elif current_pos < target_pos:
                sell_qty = max(0, passive_clip - 4)
                buy_qty = min(buy_cap, passive_clip + 2)
                ask_quote += 2

        elif regime == "short":
            if current_pos > target_pos + 3:
                buy_qty = 0
                sell_qty = min(sell_cap, passive_clip + 1)
                bid_quote -= 2
            elif current_pos > target_pos:
                buy_qty = max(0, passive_clip - 5)
                sell_qty = min(sell_cap, passive_clip)

        if current_pos >= params["hard_long"]:
            buy_qty = 0
            sell_qty = min(sell_cap, passive_clip + 4)
        elif current_pos <= -params["hard_short"]:
            sell_qty = 0
            buy_qty = min(buy_cap, passive_clip + 8)
            ask_quote += 3

        buy_qty = min(buy_qty, buy_cap)
        sell_qty = min(sell_qty, sell_cap)

        debug_info[symbol] = {
            "our_bid": bid_quote if buy_qty > 0 else None,
            "our_ask": ask_quote if sell_qty > 0 else None,
            "bid_size": buy_qty,
            "ask_size": sell_qty,
            "position": current_pos,
            "regime": regime,
            "signal": signal,
            "centre": round(centre, 3),
        }

        if buy_qty > 0:
            buy_used, sell_used = self.place_buy(
                orders, symbol, bid_quote, buy_qty, limit, pos, buy_used, sell_used
            )
        if sell_qty > 0:
            buy_used, sell_used = self.place_sell(
                orders, symbol, ask_quote, sell_qty, limit, pos, buy_used, sell_used
            )

        return buy_used, sell_used
            
    def root_signal_and_regime(self, data: Dict[str, Any]) -> Tuple[float, str]:
        symbol = "INTARIAN_PEPPER_ROOT"

        ema_fast = data["ema_fast"].get(symbol)
        ema_slow = data["ema_slow"].get(symbol)
        trend = 0.0 if ema_fast is None or ema_slow is None else (ema_fast - ema_slow)
        slope = 0.0
        hist = data["price_hist"].get(symbol, [])
        window = 15
        if len(hist) >= window:
            y = hist[-window:]
            x = list(range(window))

            x_mean = sum(x) / window
            y_mean = sum(y) / window

            num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
            den = sum((xi - x_mean) ** 2 for xi in x)

            slope = num / den if den != 0 else 0.0

        signal = trend + 0.4 * slope
        
        regime = data.get("root_regime", "neutral")
        if regime not in ("neutral", "long", "short"):
            regime = "neutral"

        # Hysteresis:
        # enter long easier than short, and require stronger confirmation to enter short
        if regime == "neutral":
            if signal > 1.2:
                regime = "long"
            elif signal < -2.0:
                regime = "short"
        elif regime == "long":
            if signal < 0.3:
                regime = "neutral"
        elif regime == "short":
            if signal > -0.5:
                regime = "neutral"

        data["root_regime"] = regime
        return signal, regime

    def carry_root(
        self,
        order_depth: OrderDepth,
        data: Dict[str, Any],
        pos: int,
        orders: List[Order],
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:

        symbol = "INTARIAN_PEPPER_ROOT"
        limit = self.LIMITS[symbol]

        mid = self.mid_price(order_depth)
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if mid is None:
            return buy_used, sell_used

        signal, regime = self.root_signal_and_regime(data)
        current_pos = pos + buy_used - sell_used

        # Asymmetric target positions:
        # more willing to be long than short
        if regime == "long":
            if signal > 3.0:
                target_pos = 70
            elif signal > 2.0:
                target_pos = 50
            elif signal > 1.2:
                target_pos = 25
            else:
                target_pos = 10
        elif regime == "short":
            if signal < -3.5:
                target_pos = -20
            elif signal < -2.5:
                target_pos = -12
            else:
                target_pos = -6
        else:
            target_pos = 0

        # LONG REGIME: accumulate and hold
        if current_pos < target_pos:
            need = target_pos - current_pos

            # Cross only if strong long signal
            if signal > 2.0 and best_ask is not None:
                qty = min(
                    need,
                    10,
                    self.remaining_buy_capacity(limit, pos, buy_used, sell_used),
                )
                if qty > 0:
                    buy_used, sell_used = self.place_buy(
                        orders, symbol, best_ask, qty, limit, pos, buy_used, sell_used
                    )

            # Passive reload bid
            if best_bid is not None:
                qty = min(
                    need,
                    8,
                    self.remaining_buy_capacity(limit, pos, buy_used, sell_used),
                )
                if qty > 0:
                    buy_used, sell_used = self.place_buy(
                        orders, symbol, best_bid + 1, qty, limit, pos, buy_used, sell_used
                    )

        # Above long target: trim gently, not aggressively
        elif regime == "long" and current_pos > target_pos:
            excess = current_pos - target_pos

            if best_ask is not None and signal < 1.0:
                qty = min(
                    excess,
                    5,
                    self.remaining_sell_capacity(limit, pos, buy_used, sell_used),
                )
                if qty > 0:
                    buy_used, sell_used = self.place_sell(
                        orders, symbol, best_ask - 1, qty, limit, pos, buy_used, sell_used
                    )

        # SHORT REGIME: shorting is allowed, but much harder / smaller
        elif regime == "short" and current_pos > target_pos:
            excess = current_pos - target_pos

            # Only cross to sell if strongly negative
            if signal < -2.8 and best_bid is not None:
                qty = min(
                    excess,
                    5,
                    self.remaining_sell_capacity(limit, pos, buy_used, sell_used),
                )
                if qty > 0:
                    buy_used, sell_used = self.place_sell(
                        orders, symbol, best_bid, qty, limit, pos, buy_used, sell_used
                    )

            # Small passive ask only for reasonably strong short signal
            elif signal < -2.0 and best_ask is not None:
                qty = min(
                    excess,
                    3,
                    self.remaining_sell_capacity(limit, pos, buy_used, sell_used),
                )
                if qty > 0:
                    buy_used, sell_used = self.place_sell(
                        orders, symbol, best_ask - 1, qty, limit, pos, buy_used, sell_used
                    )

        # If neutral and short, cover quickly
        elif regime == "neutral" and current_pos < 0:
            to_cover = -current_pos

            if best_ask is not None:
                qty = min(
                    to_cover,
                    8,
                    self.remaining_buy_capacity(limit, pos, buy_used, sell_used),
                )
                if qty > 0:
                    buy_used, sell_used = self.place_buy(
                        orders, symbol, best_ask, qty, limit, pos, buy_used, sell_used
                    )

        return buy_used, sell_used
    
    def trade_root(self, state: TradingState, data: Dict[str, Any], debug_info: Dict[str, Any]) -> List[Order]:
        if "INTARIAN_PEPPER_ROOT" not in state.order_depths:
            return []

        order_depth = state.order_depths["INTARIAN_PEPPER_ROOT"]
        pos = self.get_position(state, "INTARIAN_PEPPER_ROOT")

        params = self.PARAMS["INTARIAN_PEPPER_ROOT"]
        mid = self.mid_price(order_depth)
        if mid is not None:
            data["ema_fast"]["INTARIAN_PEPPER_ROOT"] = self.update_ema(
                data["ema_fast"].get("INTARIAN_PEPPER_ROOT"), mid, params["fast_alpha"]
            )
            data["ema_slow"]["INTARIAN_PEPPER_ROOT"] = self.update_ema(
                data["ema_slow"].get("INTARIAN_PEPPER_ROOT"), mid, params["slow_alpha"]
            )
            hist = data["price_hist"].setdefault("INTARIAN_PEPPER_ROOT", [])
            hist.append(mid)
            if len(hist) > 30:
                hist.pop(0)

        fair = self.root_centre(order_depth, data)
        if fair is not None:
            data["root_centre"]["last_centre"] = fair

        signal, regime = self.root_signal_and_regime(data)

        ema_fast = data["ema_fast"].get("INTARIAN_PEPPER_ROOT")
        ema_slow = data["ema_slow"].get("INTARIAN_PEPPER_ROOT")
        trend = 0.0 if ema_fast is None or ema_slow is None else (ema_fast - ema_slow)

        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        # Carry first if directional regime exists
        if regime in ("long", "short"):
            buy_used, sell_used = self.carry_root(
                order_depth, data, pos, orders, buy_used, sell_used
            )


        buy_used, sell_used = self.make_root(
            order_depth, data, pos, orders, buy_used, sell_used, debug_info
        )

        logger.print(
            f"ROOT pos={pos} fair={fair if fair is None else round(fair, 3)} "
            f"trend={round(trend, 3)} signal={round(signal, 3)} regime={regime} "
            f"buy_used={buy_used} sell_used={sell_used} orders={orders}"
        )
        return orders

# =================== OSMIUM =====================================
    def osmium_fair(self, order_depth: OrderDepth, data: Dict[str, Any]) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        micro = self.microprice(order_depth)
        mid = self.mid_price(order_depth)
        prev_fair = data["osmium_fair"].get("last_fair")

        anchor = 10000.0

        if micro is not None and mid is not None:
            spread = best_ask - best_bid if best_bid is not None and best_ask is not None else 8

            if spread <= 3:
                micro_w, anchor_w = 0.55, 0.45
            elif spread <= 6:
                micro_w, anchor_w = 0.45, 0.55
            else:
                micro_w, anchor_w = 0.30, 0.70

            fair = micro_w * micro + anchor_w * anchor

            if prev_fair is not None:
                fair = 0.7 * fair + 0.3 * prev_fair

            return fair

        if prev_fair is not None:
            return prev_fair

        return anchor

    
    def make_osmium(
        self,
        order_depth: OrderDepth,
        data: Dict[str, Any],
        orders: List[Order],
        pos: int,
        buy_used: int,
        sell_used: int,
        debug_info: Dict[str, Any]
    ) -> Tuple[int, int]:

        symbol = "ASH_COATED_OSMIUM"
        limit = self.LIMITS[symbol]
        params = self.PARAMS[symbol]

        fair = self.osmium_fair(order_depth, data)
        if fair is None:
            return buy_used, sell_used

        data["osmium_fair"]["last_fair"] = fair

        current_pos = pos + buy_used - sell_used
        best_bid, best_ask = self.best_bid_ask(order_depth)
        micro = self.microprice(order_depth)
        imbalance = self.book_imbalance(order_depth, levels=3)

        mid = self.mid_price(order_depth)
        hist = data["price_hist"].setdefault(symbol, [])
        if mid is not None:
            hist.append(mid)
            if len(hist) > 30:
                hist.pop(0)

        vol = self.realized_vol(hist)
        micro_dev = 0.0 if micro is None else (micro - fair)
        alpha = 0.35 * micro_dev + 0.45 * imbalance

        reservation = fair + alpha - params["inv_skew"] * current_pos

        half_spread = (
            params["typical_spread"] / 2
            + 1.2 * vol
            + 0.9 * abs(imbalance)
            + 2.0 * abs(current_pos) / limit
        )

        bid_quote = math.floor(reservation - half_spread)
        ask_quote = math.ceil(reservation + half_spread)

        # Conditional pennying only
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None

        if spread is not None and spread >= 3:
            if alpha >= -0.2 and imbalance > -0.2 and best_bid is not None:
                if reservation - (best_bid + 1) >= 0.5:
                    bid_quote = max(bid_quote, best_bid + 1)

            if alpha <= 0.2 and imbalance < 0.2 and best_ask is not None:
                if (best_ask - 1) - reservation >= 0.5:
                    ask_quote = min(ask_quote, best_ask - 1)

        if bid_quote >= ask_quote:
            centre = int(round(reservation))
            bid_quote = centre - 4
            ask_quote = centre + 4

        if bid_quote >= 10000 or ask_quote <= 10000:
            centre = 10000
            bid_quote = centre - 4
            ask_quote = centre + 4

        buy_cap = self.remaining_buy_capacity(limit, pos, buy_used, sell_used)
        sell_cap = self.remaining_sell_capacity(limit, pos, buy_used, sell_used)

        base_size = min(params["clip"], 8)
        buy_clip = min(base_size, buy_cap)
        sell_clip = min(base_size, sell_cap)

        # Early inventory management
        if current_pos > 10:
            buy_clip = max(0, buy_clip - 2)
            sell_clip = min(sell_cap, sell_clip + 1)
            bid_quote -= 1
        elif current_pos < -10:
            sell_clip = max(0, sell_clip - 2)
            buy_clip = min(buy_cap, buy_clip + 1)
            ask_quote += 1

        if current_pos > 25:
            buy_clip = 0
            sell_clip = min(sell_cap, sell_clip + 2)
        elif current_pos < -25:
            sell_clip = 0
            buy_clip = min(buy_cap, buy_clip + 2)

        if current_pos >= params["hard_pos"]:
            buy_clip = 0
            sell_clip = min(sell_cap, max(sell_clip, params["clip"]))
        elif current_pos <= -params["hard_pos"]:
            sell_clip = 0
            buy_clip = min(buy_cap, max(buy_clip, params["clip"]))

        if buy_clip > 0:
            buy_used, sell_used = self.place_buy(
                orders, symbol, bid_quote, buy_clip, limit, pos, buy_used, sell_used
            )
        if sell_clip > 0:
            buy_used, sell_used = self.place_sell(
                orders, symbol, ask_quote, sell_clip, limit, pos, buy_used, sell_used
            )

        debug_info[symbol] = {
            "our_bid": bid_quote if buy_clip > 0 else None,
            "our_ask": ask_quote if sell_clip > 0 else None,
            "bid_size": buy_clip,
            "ask_size": sell_clip,
            "position": current_pos,
            "fair": round(fair, 3),
            "signal": round(alpha, 3),
            "imbalance": round(imbalance, 3),
            "vol": round(vol, 3),
            "reservation": round(reservation, 3),
        }

        return buy_used, sell_used

    def take_osmium(
        self,
        order_depth: OrderDepth,
        data: Dict[str, Any],
        orders: List[Order],
        pos: int,
        buy_used: int,
        sell_used: int
    ) -> Tuple[int, int]:

        symbol = "ASH_COATED_OSMIUM"
        limit = self.LIMITS[symbol]
        params = self.PARAMS[symbol]

        fair = self.osmium_fair(order_depth, data)
        if fair is None:
            return buy_used, sell_used

        data["osmium_fair"]["last_fair"] = fair

        micro = self.microprice(order_depth)
        imbalance = self.book_imbalance(order_depth, levels=3)
        current_pos = pos + buy_used - sell_used

        mid = self.mid_price(order_depth)
        hist = data["price_hist"].setdefault(symbol, [])
        if mid is not None:
            hist.append(mid)
            if len(hist) > 30:
                hist.pop(0)

        vol = self.realized_vol(hist)
        micro_dev = 0.0 if micro is None else (micro - fair)
        alpha = 0.35 * micro_dev + 0.45 * imbalance

        buy_threshold = fair + alpha - (
            params["take_edge"] + 0.5 * vol + 0.6 * max(0, current_pos) / limit
        )
        sell_threshold = fair + alpha + (
            params["take_edge"] + 0.5 * vol + 0.6 * max(0, -current_pos) / limit
        )

        for ask_price, ask_qty in self.sorted_asks(order_depth):
            ask_qty = -ask_qty
            if ask_price <= buy_threshold:
                edge = (fair + alpha) - ask_price
                qty = min(
                    ask_qty,
                    self.remaining_buy_capacity(limit, pos, buy_used, sell_used),
                    max(2, min(params["clip"], int(3 + 2 * edge)))
                )
                if qty <= 0:
                    break
                buy_used, sell_used = self.place_buy(
                    orders, symbol, ask_price, qty, limit, pos, buy_used, sell_used
                )
            else:
                break

        for bid_price, bid_qty in self.sorted_bids(order_depth):
            if bid_price >= sell_threshold:
                edge = bid_price - (fair + alpha)
                qty = min(
                    bid_qty,
                    self.remaining_sell_capacity(limit, pos, buy_used, sell_used),
                    max(2, min(params["clip"], int(3 + 2 * edge)))
                )
                if qty <= 0:
                    break
                buy_used, sell_used = self.place_sell(
                    orders, symbol, bid_price, qty, limit, pos, buy_used, sell_used
                )
            else:
                break

        return buy_used, sell_used
    
    def trade_osmium(self, state: TradingState, data: Dict[str, Any], debug_info: Dict[str, Any]) -> List[Order]:
        if "ASH_COATED_OSMIUM" not in state.order_depths:
            return []

        order_depth = state.order_depths["ASH_COATED_OSMIUM"]
        pos = self.get_position(state, "ASH_COATED_OSMIUM")
        fair = self.osmium_fair(order_depth, data)

        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        buy_used, sell_used = self.take_osmium(
            order_depth, data, orders, pos, buy_used, sell_used
        )
        buy_used, sell_used = self.make_osmium(
            order_depth, data, orders, pos, buy_used, sell_used, debug_info
        )

        logger.print(
            f"OSMIUM pos={pos} fair={fair if fair is None else round(fair, 3)}"
            f"buy_used={buy_used} sell_used={sell_used} orders={orders}"
        )
        return orders

    # ============================ RUN ==========================


    def run(self, state: TradingState):
        data = self.load_data(state.traderData)
        result: Dict[Symbol, List[Order]] = {}
        debug_info: Dict[str, Any] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self.trade_osmium(state, data, debug_info)

        #if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            #result["INTARIAN_PEPPER_ROOT"] = self.trade_root(state, data, debug_info)

        trader_data = self.save_data(data)
        conversions = 0
        logger.flush(state, result, conversions, trader_data, debug_info)
        return result, conversions, trader_data