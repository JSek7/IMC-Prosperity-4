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
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
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

    PARAMS = {"INTARIAN_PEPPER_ROOT": {
        "take_edge": 0.0,
        "typical_spread": 8.0,
        "clip": 15,
        "inv_skew": 0.10,
        "hard_pos": 70,
        "wide_spread": 8,
        "target_scale": 12,
        "fast_alpha": 0.20,
        "slow_alpha": 0.05,
        "trend_mult": 0.2,
},
        "ASH_COATED_OSMIUM": {
            "take_edge": 1.0,
            "typical_spread": 8.0,
            "clip": 20,
            "vol_mult": 1.2,
            "inv_skew": 0.18,
            "imbalance_mult": 0.5,
            "hard_pos": 65,
            "fast_alpha": 0.24,
            "slow_alpha": 0.07,
            "trend_mult": 0.50,
        },
    }

# Helper functions for data manipulation and strategy logic

    def load_data(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {"ema_fast": {}, "ema_slow": {}, "price_hist": {}, "osmium_fair": {}, "root_centre": {}}

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
    ) -> Tuple[int, int]:

        symbol = "INTARIAN_PEPPER_ROOT"
        params = self.PARAMS[symbol]
        limit = self.LIMITS[symbol]

        ema_fast = data["ema_fast"].get(symbol)
        ema_slow = data["ema_slow"].get(symbol)
        trend = 0.0
        if ema_fast is not None and ema_slow is not None:
            trend = ema_fast - ema_slow

        centre = self.root_centre(order_depth, data)
        best_bid, best_ask = self.best_bid_ask(order_depth)

        if centre is None:
            return buy_used, sell_used

        current_pos = pos + buy_used - sell_used
        pos_skew = params["inv_skew"] * current_pos

        spread = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        
        

        # Liquid book — quote inside the spread to get fills
        if best_bid is not None and best_ask is not None and spread is not None and spread <= params["wide_spread"]:
            bid_quote = best_bid + 1
            ask_quote = best_ask - 1

            # If spread is only 2, can't improve both sides — just join
            if bid_quote >= ask_quote:
                bid_quote = best_bid
                ask_quote = best_ask

        # Thin/wide/one-sided book — use centre-based quoting
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

        # Apply inventory skew to both branches
        bid_quote = math.floor(bid_quote - pos_skew)
        ask_quote = math.ceil(ask_quote - pos_skew)

        if bid_quote >= ask_quote:
            c = int(round(centre))
            bid_quote = c - 1
            ask_quote = c + 1

        buy_cap = self.remaining_buy_capacity(limit, pos, buy_used, sell_used)
        sell_cap = self.remaining_sell_capacity(limit, pos, buy_used, sell_used)

        passive_clip = max(1, params["clip"] // 2)

        target_pos = max(-limit, min(limit, int(trend * params["target_scale"])))

        # Always quote both sides — use target to skew SIZE not existence
        buy_qty = passive_clip
        sell_qty = passive_clip

        # Skew sizes toward target position
        if current_pos < target_pos:
            sell_qty = max(0, passive_clip - 2)
        elif current_pos > target_pos:
            buy_qty = max(0, passive_clip - 2)

        # Hard position limits
        if current_pos >= params["hard_pos"]:
            buy_qty = 0
            sell_qty = min(sell_cap, passive_clip + 4)
        elif current_pos <= -params["hard_pos"]:
            sell_qty = 0
            buy_qty = min(buy_cap, passive_clip + 4)

        buy_qty = min(buy_qty, buy_cap)
        sell_qty = min(sell_qty, sell_cap)

        if buy_qty > 0:
            buy_used, sell_used = self.place_buy(
                orders, symbol, bid_quote, buy_qty, limit, pos, buy_used, sell_used
            )
        if sell_qty > 0:
            buy_used, sell_used = self.place_sell(
                orders, symbol, ask_quote, sell_qty, limit, pos, buy_used, sell_used
            )
                
    
        return buy_used, sell_used
    

    def momentum_root(self, order_depth, data, pos, orders, buy_used, sell_used):

        symbol = "INTARIAN_PEPPER_ROOT"
        limit = self.LIMITS[symbol]
        params = self.PARAMS[symbol]


        mid = self.mid_price(order_depth)
        if mid is None:
            return buy_used, sell_used

        best_bid, best_ask = self.best_bid_ask(order_depth)

        # --- update EMAs ---
        ema_fast = data["ema_fast"].get(symbol)
        ema_slow = data["ema_slow"].get(symbol)

        if ema_fast is None or ema_slow is None:
            return buy_used, sell_used

        hist = data["price_hist"].get(symbol, [])
        slope = 0.0
        if len(hist) >= 6:
            slope = mid - hist[-6]

        signal = (ema_fast - ema_slow) + 0.2 * slope
        current_pos = pos + buy_used - sell_used

        # --- target position ---
        target_pos = int(signal * 8)
        target_pos = max(-limit, min(limit, target_pos))

        # --- BUY LOGIC ---
        if current_pos < target_pos and best_ask is not None:

            need = target_pos - current_pos

            # buy dip
            if mid < ema_fast:
                qty = min(10, need,
                        self.remaining_buy_capacity(limit, pos, buy_used, sell_used))

                if qty > 0:
                    buy_used, sell_used = self.place_buy(
                        orders, symbol, best_ask, qty, limit, pos, buy_used, sell_used
                    )

            # strong trend → don't wait
            elif signal > 2.0:
                qty = min(10, need,
                        self.remaining_buy_capacity(limit, pos, buy_used, sell_used))

                if qty > 0:
                    buy_used, sell_used = self.place_buy(
                        orders, symbol, best_ask, qty, limit, pos, buy_used, sell_used
                    )

        # --- SELL LOGIC ---
        elif current_pos > target_pos and best_bid is not None:

            excess = current_pos - target_pos

            qty = min(10, excess,
                    self.remaining_sell_capacity(limit, pos, buy_used, sell_used))

            if qty > 0:
                buy_used, sell_used = self.place_sell(
                    orders, symbol, best_bid, qty, limit, pos, buy_used, sell_used
                )

        return buy_used, sell_used

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
        params = self.PARAMS[symbol]

        mid = self.mid_price(order_depth)
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if mid is None:
            return buy_used, sell_used

        ema_fast = data["ema_fast"].get(symbol)
        ema_slow = data["ema_slow"].get(symbol)
        if ema_fast is None or ema_slow is None:
            return buy_used, sell_used

        trend = ema_fast - ema_slow
        hist = data["price_hist"].get(symbol, [])
        slope = 0.0
        if len(hist) >= 5:
            slope = mid - hist[-5]

        signal = trend + 0.25 * slope
        current_pos = pos + buy_used - sell_used

        # regime-based target
        if signal > 3.0:
            target_pos = 70
        elif signal > 2.0:
            target_pos = 50
        elif signal > 1.0:
            target_pos = 25
        elif signal > 0.4:
            target_pos = 10
        elif signal < -3.5:
            target_pos = -20
        elif signal < -2.0:
            target_pos = -12
        elif signal < -1.0:
            target_pos = -6
        else:
            target_pos = 0

        # UPTREND: accumulate and hold
        if current_pos < target_pos:
            need = target_pos - current_pos

            # cross only if strong signal
            if signal > 2.0 and best_ask is not None:
                qty = min(need, 10, self.remaining_buy_capacity(limit, pos, buy_used, sell_used))
                if qty > 0:
                    buy_used, sell_used = self.place_buy(
                        orders, symbol, best_ask, qty, limit, pos, buy_used, sell_used
                    )

            # passive bid for pullback accumulation
            if best_bid is not None:
                qty = min(need, 8, self.remaining_buy_capacity(limit, pos, buy_used, sell_used))
                if qty > 0:
                    buy_used, sell_used = self.place_buy(
                        orders, symbol, best_bid + 1, qty, limit, pos, buy_used, sell_used
                    )

        # DOWNTREND: symmetric short logic
        elif current_pos > target_pos:
            excess = current_pos - target_pos

            if signal < -2.0 and best_bid is not None:
                qty = min(excess, 10, self.remaining_sell_capacity(limit, pos, buy_used, sell_used))
                if qty > 0:
                    buy_used, sell_used = self.place_sell(
                        orders, symbol, best_bid, qty, limit, pos, buy_used, sell_used
                    )

            if best_ask is not None:
                qty = min(excess, 8, self.remaining_sell_capacity(limit, pos, buy_used, sell_used))
                if qty > 0:
                    buy_used, sell_used = self.place_sell(
                        orders, symbol, best_ask - 1, qty, limit, pos, buy_used, sell_used
                    )

        return buy_used, sell_used

    def trade_root(self, state: TradingState, data: Dict[str, Any]) -> List[Order]:
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
            if len(hist) > 20:
                hist.pop(0)


        fair = self.root_centre(order_depth, data)
        if fair is not None:
            data["root_centre"]["last_centre"] = fair

        ema_fast = data["ema_fast"].get("INTARIAN_PEPPER_ROOT")
        ema_slow = data["ema_slow"].get("INTARIAN_PEPPER_ROOT")
        trend = 0.0 if ema_fast is None or ema_slow is None else (ema_fast - ema_slow)

        best_bid, best_ask = self.best_bid_ask(order_depth)
        spread = None if best_bid is None or best_ask is None else (best_ask - best_bid)

        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        strong_trend = abs(trend) > 1.0
        tight_book = spread is not None and spread <= 5

        if strong_trend:
            buy_used, sell_used = self.carry_root(order_depth, data, pos, orders, buy_used, sell_used)

            # only add passive MM if signal is not too strong
            if abs(trend) < 2.5:
                buy_used, sell_used = self.make_root(order_depth, data, pos, orders, buy_used, sell_used)
        else:
            buy_used, sell_used = self.make_root(order_depth, data, pos, orders, buy_used, sell_used)

        logger.print(
            f"ROOT pos={pos} fair={fair if fair is None else round(fair, 3)} "
            f"trend={round(trend, 3)} buy_used={buy_used} sell_used={sell_used} orders={orders}"
        )
        return orders

# =================== OSMIUM =====================================
 
    def osmium_fair(self, order_depth: OrderDepth, data: Dict[str, Any]) -> Optional[float]:
        params = self.PARAMS["ASH_COATED_OSMIUM"]

        best_bid, best_ask = self.best_bid_ask(order_depth)
        micro = self.microprice(order_depth)
        mid = self.mid_price(order_depth)
        imbalance = self.book_imbalance(order_depth, levels=3)

        prev_fair = data["osmium_fair"].get("last_fair")

        # Full book available
        if micro is not None and mid is not None:
            anchor = prev_fair if prev_fair is not None else mid

            fair = (
                0.2 * micro
                + 0.1 * mid
                + 0.7 * anchor
                #+ params["imbalance_mult"] * 0.2 * imbalance
            )
            return fair

        # One-sided or sparse book fallback
        if prev_fair is not None:
            if best_bid is not None and best_ask is None:
                return 0.9 * prev_fair + 0.1 * float(best_bid)
            if best_ask is not None and best_bid is None:
                return 0.9 * prev_fair + 0.1 * float(best_ask)
            return prev_fair

        # Final fallback if no previous fair exists
        if best_bid is not None:
            return float(best_bid) - params["typical_spread"]//2
        if best_ask is not None:
            return float(best_ask) + params["typical_spread"]//2

        return None
    

    def take_osmium(
            self, 
            order_depth: OrderDepth, 
            data: Dict[str, Any],
            orders: List[Order],
            pos: int,
            buy_used: int,
            sell_used: int
        ) -> Tuple[int, int]:   


        limit = self.LIMITS["ASH_COATED_OSMIUM"]
        params = self.PARAMS["ASH_COATED_OSMIUM"]
        fair = self.osmium_fair(order_depth, data)
        if fair is not None:
            data["osmium_fair"]["last_fair"] = fair
        elif fair is None:
            return buy_used, sell_used
        
        # if ask price is sufficiently below fair or at fair with a long position, buy
        for ask_price, ask_qty in self.sorted_asks(order_depth):
            ask_qty = -ask_qty
            current_pos = pos + buy_used - sell_used

            if ask_price <= fair - params["take_edge"]:
                qty = min(ask_qty, self.remaining_buy_capacity(limit, pos, buy_used, sell_used), params["clip"])
                if qty <= 0:
                     break
                buy_used, sell_used = self.place_buy(
                    orders, "ASH_COATED_OSMIUM", ask_price, qty, limit, pos, buy_used, sell_used
                )
            else:
                break

        # if bid price is sufficiently above fair or at fair with a short position, sell
        for bid_price, bid_qty in self.sorted_bids(order_depth):
            current_pos = pos + buy_used - sell_used

            if bid_price > fair + params["take_edge"]:
                qty = min(bid_qty, self.remaining_sell_capacity(limit, pos, buy_used, sell_used), params["clip"])
                if qty <= 0:
                    break
                buy_used, sell_used = self.place_sell(
                    orders, "ASH_COATED_OSMIUM", bid_price, qty, limit, pos, buy_used, sell_used
                )
            else:
                break

        return buy_used, sell_used
    

    def make_osmium(
            self, 
            order_depth: OrderDepth, 
            data: Dict[str, Any],
            orders: List[Order],
            pos: int,
            buy_used: int,
            sell_used: int
        ) -> Tuple[int, int]:
    
        limit = self.LIMITS["ASH_COATED_OSMIUM"]
        params = self.PARAMS["ASH_COATED_OSMIUM"]
        fair = self.osmium_fair(order_depth, data)
        data["osmium_fair"]["last_fair"] = fair
        best_bid, best_ask = self.best_bid_ask(order_depth)

        #calculate position skew
        current_pos = pos + buy_used - sell_used
        pos_skew = params["inv_skew"] * current_pos

    
        if fair is None:
            return buy_used, sell_used
        
        # calculate own spread anchor based on volatility/skew/imbalance

        if best_bid is not None:
            bid_quote = best_bid + 1  # penny in front
        else:
            bid_quote = math.floor(fair - params["typical_spread"] / 2 - pos_skew)

        if best_ask is not None:
            ask_quote = best_ask - 1  # penny in front
        else:
            ask_quote = math.ceil(fair + params["typical_spread"] / 2 - pos_skew)

        # logic if no ask/no bid --> be less aggressive on the side with no liquidity, but anchor to fair

        if best_ask is None and best_bid is not None:
            ask_quote = ask_quote + 1
        if best_bid is None and best_ask is not None:
            bid_quote = bid_quote - 1

        # reset anchors if they are on the wrong side of the book

        if bid_quote >= ask_quote:
            centre = int(round(fair))
            bid_quote = centre - 1
            ask_quote = centre + 1
            
        # if current pos is high but not above hard_pos --> increase size on the side that would reduce position and quote more towards fair

        buy_cap = self.remaining_buy_capacity(limit, pos, buy_used, sell_used)
        sell_cap = self.remaining_sell_capacity(limit, pos, buy_used, sell_used)

        buy_clip = min(params["clip"], buy_cap)
        sell_clip = min(params["clip"], sell_cap)

        if current_pos > 18:
            buy_clip = max(0, buy_clip - current_pos // 10)
        elif current_pos < -18:
            sell_clip = max(0, sell_clip - (-current_pos) // 10)


        # if current pos is above hard_pos --> quote on sided and aggressively towards fair

        if current_pos >= params["hard_pos"]:
            buy_clip = 0
            sell_clip = min(sell_cap, max(sell_clip, params["clip"] + 5))
        elif current_pos <= -params["hard_pos"]:
            sell_clip = 0
            buy_clip = min(buy_cap, max(buy_clip, params["clip"] + 5))

        if buy_clip > 0:
            buy_used, sell_used = self.place_buy(
                orders, "ASH_COATED_OSMIUM", bid_quote, buy_clip, limit, pos, buy_used, sell_used
            )
        if sell_clip > 0:
            buy_used, sell_used = self.place_sell(
                orders, "ASH_COATED_OSMIUM", ask_quote, sell_clip, limit, pos, buy_used, sell_used
            )

        return buy_used, sell_used
    

    def trade_osmium(self, state: TradingState, data: Dict[str, Any]) -> List[Order]:
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
            order_depth, data, orders, pos, buy_used, sell_used
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

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result["ASH_COATED_OSMIUM"] = self.trade_osmium(state, data)

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result["INTARIAN_PEPPER_ROOT"] = self.trade_root(state, data)

        trader_data = self.save_data(data)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data