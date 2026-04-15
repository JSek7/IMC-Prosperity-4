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
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    PARAMS = {
        "EMERALDS": {
            "take_edge": 0.0,
            "mm_edge": 3.0,
            "clip": 18,
            "inv_skew": 0.12,
            "hard_pos": 62,
        },
        "TOMATOES": {
            "take_edge": 2.0,
            "mm_edge": 3.0,
            "clip": 7,
            "vol_mult": 1.2,
            "inv_skew": 0.28,
            "imbalance_mult": 1.5,
            "hard_pos": 52,
            "fast_alpha": 0.24,
            "slow_alpha": 0.07,
            "trend_mult": 0.95,
        },
    }

    def load_data(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return {"ema_fast": {}, "ema_slow": {}, "price_hist": {}}

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

    # -------------------- EMERALDS --------------------

    def emerald_fair(self) -> float:
        return 10000.0

    def take_emeralds(
        self,
        order_depth: OrderDepth,
        pos: int,
        orders: List[Order],
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:
        limit = self.LIMITS["EMERALDS"]
        fair = self.emerald_fair()

        for ask_price, ask_qty in self.sorted_asks(order_depth):
            ask_qty = -ask_qty
            current_pos = pos + buy_used - sell_used

            if ask_price < fair or (ask_price == fair and current_pos < 0):
                qty = min(ask_qty, self.remaining_buy_capacity(limit, pos, buy_used, sell_used))
                if qty <= 0:
                    break
                buy_used, sell_used = self.place_buy(
                    orders, "EMERALDS", ask_price, qty, limit, pos, buy_used, sell_used
                )
            else:
                break

        for bid_price, bid_qty in self.sorted_bids(order_depth):
            current_pos = pos + buy_used - sell_used

            if bid_price > fair or (bid_price == fair and current_pos > 0):
                qty = min(bid_qty, self.remaining_sell_capacity(limit, pos, buy_used, sell_used))
                if qty <= 0:
                    break
                buy_used, sell_used = self.place_sell(
                    orders, "EMERALDS", bid_price, qty, limit, pos, buy_used, sell_used
                )
            else:
                break

        return buy_used, sell_used

    def make_emeralds(
        self,
        order_depth: OrderDepth,
        pos: int,
        orders: List[Order],
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:
        params = self.PARAMS["EMERALDS"]
        limit = self.LIMITS["EMERALDS"]
        fair = self.emerald_fair()
        best_bid, best_ask = self.best_bid_ask(order_depth)

        current_pos = pos + buy_used - sell_used
        skew = params["inv_skew"] * current_pos

        bid_anchor = fair - params["mm_edge"] - skew
        ask_anchor = fair + params["mm_edge"] - skew

        bid_candidates = [p for p in order_depth.buy_orders.keys() if p < bid_anchor]
        ask_candidates = [p for p in order_depth.sell_orders.keys() if p > ask_anchor]

        bid_quote = (max(bid_candidates) + 1) if bid_candidates else math.floor(bid_anchor)
        ask_quote = (min(ask_candidates) - 1) if ask_candidates else math.ceil(ask_anchor)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - 1)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + 1)

        if bid_quote >= ask_quote:
            center = int(round(fair - skew))
            bid_quote = center - 1
            ask_quote = center + 1

        buy_cap = self.remaining_buy_capacity(limit, pos, buy_used, sell_used)
        sell_cap = self.remaining_sell_capacity(limit, pos, buy_used, sell_used)

        buy_clip = min(params["clip"], buy_cap)
        sell_clip = min(params["clip"], sell_cap)

        if current_pos > 18:
            buy_clip = max(0, buy_clip - current_pos // 10)
        elif current_pos < -18:
            sell_clip = max(0, sell_clip - (-current_pos) // 10)

        if current_pos >= params["hard_pos"]:
            buy_clip = 0
            sell_clip = min(sell_cap, max(sell_clip, params["clip"] + 5))
        elif current_pos <= -params["hard_pos"]:
            sell_clip = 0
            buy_clip = min(buy_cap, max(buy_clip, params["clip"] + 5))

        if buy_clip > 0:
            buy_used, sell_used = self.place_buy(
                orders, "EMERALDS", bid_quote, buy_clip, limit, pos, buy_used, sell_used
            )
        if sell_clip > 0:
            buy_used, sell_used = self.place_sell(
                orders, "EMERALDS", ask_quote, sell_clip, limit, pos, buy_used, sell_used
            )

        return buy_used, sell_used

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        if "EMERALDS" not in state.order_depths:
            return []

        order_depth = state.order_depths["EMERALDS"]
        pos = self.get_position(state, "EMERALDS")
        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        buy_used, sell_used = self.take_emeralds(order_depth, pos, orders, buy_used, sell_used)
        buy_used, sell_used = self.make_emeralds(order_depth, pos, orders, buy_used, sell_used)

        logger.print(
            f"EMERALDS pos={pos} buy_used={buy_used} sell_used={sell_used} orders={orders}"
        )
        return orders

    # -------------------- TOMATOES --------------------

    def book_imbalance(self, order_depth: OrderDepth, levels: int = 3) -> float:
        bids = self.sorted_bids(order_depth)[:levels]
        asks = self.sorted_asks(order_depth)[:levels]

        bid_vol = sum(qty for _, qty in bids)
        ask_vol = sum(-qty for _, qty in asks)

        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    def tomatoes_fair_and_trend(
        self,
        order_depth: OrderDepth,
        data: Dict[str, Any],
    ) -> Tuple[float, float]:
        params = self.PARAMS["TOMATOES"]
        raw = self.microprice(order_depth)
        if raw is None:
            raw = self.mid_price(order_depth)

        #if we have no price data, just return previous fair and trend (or 0 if we have no history) - better to keep trading on the trend than to stop completely
        if raw is None:
            fast = data["ema_fast"].get("TOMATOES")
            slow = data["ema_slow"].get("TOMATOES")
            if fast is None and slow is None:
                return 0.0, 0.0
            if fast is None:
                fast = slow
            if slow is None:
                slow = fast
            trend = fast - slow
            fair = fast + params["trend_mult"] * trend
            return fair, trend

        if raw is not None:
            hist = data["price_hist"].get("TOMATOES", [])
            hist.append(raw)

            max_hist = 20
            if len(hist) > max_hist:
                hist = hist[-max_hist:]

            data["price_hist"]["TOMATOES"] = hist


        prev_fast = data["ema_fast"].get("TOMATOES")
        prev_slow = data["ema_slow"].get("TOMATOES")

        fast = self.update_ema(prev_fast, raw, params["fast_alpha"])
        slow = self.update_ema(prev_slow, raw, params["slow_alpha"])

        data["ema_fast"]["TOMATOES"] = fast
        data["ema_slow"]["TOMATOES"] = slow

        trend = fast - slow
        fair = fast + params["trend_mult"] * trend
        return fair, trend

    def take_tomatoes(
        self,
        order_depth: OrderDepth,
        fair: float,
        trend: float,
        pos: int,
        orders: List[Order],
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:
        params = self.PARAMS["TOMATOES"]
        limit = self.LIMITS["TOMATOES"]

        for ask_price, ask_qty in self.sorted_asks(order_depth):
            ask_qty = -ask_qty
            current_pos = pos + buy_used - sell_used

            cheap = ask_price <= math.floor(fair - params["take_edge"])
            cover_short = current_pos < 0 and ask_price <= math.floor(fair - 1)
            trend_aligned_buy = trend > 0.8 and ask_price <= math.floor(fair - 1)

            if not (cheap or cover_short or trend_aligned_buy):
                break

            qty = min(ask_qty, self.remaining_buy_capacity(limit, pos, buy_used, sell_used))
            if qty <= 0:
                break

            buy_used, sell_used = self.place_buy(
                orders, "TOMATOES", ask_price, qty, limit, pos, buy_used, sell_used
            )

        for bid_price, bid_qty in self.sorted_bids(order_depth):
            current_pos = pos + buy_used - sell_used

            rich = bid_price >= math.ceil(fair + params["take_edge"])
            cover_long = current_pos > 0 and bid_price >= math.ceil(fair + 1)
            trend_aligned_sell = trend < -0.8 and bid_price >= math.ceil(fair + 1)

            if not (rich or cover_long or trend_aligned_sell):
                break

            qty = min(bid_qty, self.remaining_sell_capacity(limit, pos, buy_used, sell_used))
            if qty <= 0:
                break

            buy_used, sell_used = self.place_sell(
                orders, "TOMATOES", bid_price, qty, limit, pos, buy_used, sell_used
            )

        return buy_used, sell_used

    def make_tomatoes(
        self,
        order_depth: OrderDepth,
        fair: float,
        trend: float,
        pos: int,
        orders: List[Order],
        buy_used: int,
        sell_used: int,
        data: Dict[str, Any],
    ) -> Tuple[int, int]:
        params = self.PARAMS["TOMATOES"]
        limit = self.LIMITS["TOMATOES"]
        best_bid, best_ask = self.best_bid_ask(order_depth)

        current_pos = pos + buy_used - sell_used
        skew = params["inv_skew"] * current_pos

        hist = data["price_hist"].get("TOMATOES", [])
        vol = self.realized_vol(hist) if hist else 0.0

        trend_edge = min(1.5, 0.6 * abs(trend))
        vol_edge = min(2.0, params["vol_mult"] * vol)

        dynamic_edge = params["mm_edge"] + trend_edge + vol_edge

        #add order imbalance to reservation price, and increase edge when imbalance is high - more incentive to lean into the move and provide liquidity on that side
        dynamic_edge = params["mm_edge"] + min(1.5, 0.6 * abs(trend))
        imbalance = self.book_imbalance(order_depth, levels=3)

        reservation = fair - skew + params["imbalance_mult"]*imbalance

        bid_anchor = reservation - dynamic_edge
        ask_anchor = reservation + dynamic_edge

        bid_candidates = [p for p in order_depth.buy_orders.keys() if p < bid_anchor]
        ask_candidates = [p for p in order_depth.sell_orders.keys() if p > ask_anchor]

        bid_quote = (max(bid_candidates) + 1) if bid_candidates else math.floor(bid_anchor)
        ask_quote = (min(ask_candidates) - 1) if ask_candidates else math.ceil(ask_anchor)

        if best_ask is not None:
            bid_quote = min(bid_quote, best_ask - 1)
        if best_bid is not None:
            ask_quote = max(ask_quote, best_bid + 1)

        if bid_quote >= ask_quote:
            center = int(round(reservation))
            bid_quote = center - 1
            ask_quote = center + 1

        buy_cap = self.remaining_buy_capacity(limit, pos, buy_used, sell_used)
        sell_cap = self.remaining_sell_capacity(limit, pos, buy_used, sell_used)

        buy_clip = min(params["clip"], buy_cap)
        sell_clip = min(params["clip"], sell_cap)

        # Trend gating: quote less on the wrong side of the move
        if trend < -1.2:
            buy_clip = max(0, buy_clip - 5)
            sell_clip = min(sell_cap, sell_clip + 3)
        elif trend < -0.6:
            buy_clip = max(0, buy_clip - 2)
            sell_clip = min(sell_cap, sell_clip + 1)
        elif trend > 1.2:
            sell_clip = max(0, sell_clip - 5)
            buy_clip = min(buy_cap, buy_clip + 3)
        elif trend > 0.6:
            sell_clip = max(0, sell_clip - 2)
            buy_clip = min(buy_cap, buy_clip + 1)

        # Inventory + trend disagreement => unwind harder
        if current_pos > 0 and trend < -0.6:
            buy_clip = 0
            sell_clip = min(sell_cap, max(sell_clip, params["clip"] + 4))
        elif current_pos < 0 and trend > 0.6:
            sell_clip = 0
            buy_clip = min(buy_cap, max(buy_clip, params["clip"] + 4))

        if current_pos > 14:
            buy_clip = max(0, buy_clip - current_pos // 10)
        elif current_pos < -14:
            sell_clip = max(0, sell_clip - (-current_pos) // 10)

        if current_pos >= params["hard_pos"]:
            buy_clip = 0
            sell_clip = min(sell_cap, max(sell_clip, params["clip"] + 5))
        elif current_pos <= -params["hard_pos"]:
            sell_clip = 0
            buy_clip = min(buy_cap, max(buy_clip, params["clip"] + 5))

        if buy_clip > 0:
            buy_used, sell_used = self.place_buy(
                orders, "TOMATOES", bid_quote, buy_clip, limit, pos, buy_used, sell_used
            )
        if sell_clip > 0:
            buy_used, sell_used = self.place_sell(
                orders, "TOMATOES", ask_quote, sell_clip, limit, pos, buy_used, sell_used
            )

        return buy_used, sell_used

    def trade_tomatoes(self, state: TradingState, data: Dict[str, Any]) -> List[Order]:
        if "TOMATOES" not in state.order_depths:
            return []

        order_depth = state.order_depths["TOMATOES"]
        pos = self.get_position(state, "TOMATOES")
        fair, trend = self.tomatoes_fair_and_trend(order_depth, data)

        orders: List[Order] = []
        buy_used = 0
        sell_used = 0

        buy_used, sell_used = self.take_tomatoes(
            order_depth, fair, trend, pos, orders, buy_used, sell_used
        )
        buy_used, sell_used = self.make_tomatoes(
            order_depth, fair, trend, pos, orders, buy_used, sell_used, data
        )

        logger.print(
            f"TOMATOES pos={pos} fair={round(fair, 3)} trend={round(trend, 3)} "
            f"buy_used={buy_used} sell_used={sell_used} orders={orders}"
        )
        return orders

    # -------------------- RUN --------------------

    def run(self, state: TradingState):
        data = self.load_data(state.traderData)
        result: Dict[Symbol, List[Order]] = {}

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state)

        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.trade_tomatoes(state, data)

        trader_data = self.save_data(data)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data