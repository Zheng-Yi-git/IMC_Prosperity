import json
import math
from collections import deque
from enum import Enum
from json import JSONEncoder
from statistics import NormalDist
from typing import Any, Dict, List

import jsonpickle
import numpy as np

INF = 1e9
normalDist = NormalDist(0, 1)


# ---------------------------------- datamodel.py -----------------------------------
# --------------------- modified by Yi Zheng ----------------------------------------


Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )

    def __repr__(self) -> str:
        return (
            "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
        )


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class ConversionObservation:
    def __init__(
        self,
        bidPrice: float,
        askPrice: float,
        transportFees: float,
        exportTariff: float,
        importTariff: float,
        sugarPrice: float,
        sunlightIndex: float,
    ):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex


class Observation:
    def __init__(
        self,
        plainValueObservations: Dict[Product, ObservationValue],
        conversionObservations: Dict[Product, ConversionObservation],
    ) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return (
            "(plainValueObservations: "
            + jsonpickle.encode(self.plainValueObservations)
            + ", conversionObservations: "
            + jsonpickle.encode(self.conversionObservations)
            + ")"
        )


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class Trade:
    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: UserId = None,
        seller: UserId = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + self.buyer
            + " << "
            + self.seller
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )

    def __repr__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + self.buyer
            + " << "
            + self.seller
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class TradingState(object):
    def __init__(
        self,
        traderData: str,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Observation,
    ):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


# ---------------------------------- datamodel.py -----------------------------------
# --------------------- modified by Yi Zheng ----------------------------------------


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        print("debugging")

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
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

        return value[: max_length - 3] + "..."


logger = Logger()


class Status:

    _position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
        "CROISSANTS": 250,
        "JAMS": 350,
        "DJEMBES": 60,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
    }

    _state = None

    _realtime_position = {key: 0 for key in _position_limit.keys()}

    _hist_order_depths = {
        product: {
            "bidprc1": [],
            "bidamt1": [],
            "bidprc2": [],
            "bidamt2": [],
            "bidprc3": [],
            "bidamt3": [],
            "askprc1": [],
            "askamt1": [],
            "askprc2": [],
            "askamt2": [],
            "askprc3": [],
            "askamt3": [],
        }
        for product in _position_limit.keys()
    }

    _hist_observation = {
        "sunlight": [],
        "humidity": [],
        "transportFees": [],
        "exportTariff": [],
        "importTariff": [],
        "bidPrice": [],
        "askPrice": [],
    }

    _num_data = 0

    def __init__(self, product: str) -> None:
        """Initialize status object.

        Args:
            product (str): product

        """
        self.product = product

    @classmethod
    def cls_update(cls, state: TradingState) -> None:
        """Update trading state.

        Args:
            state (TradingState): trading state

        """
        # Update TradingState
        cls._state = state
        # Update realtime position
        for product, posit in state.position.items():
            cls._realtime_position[product] = posit
        # Update historical order_depths
        for product, orderdepth in state.order_depths.items():
            cnt = 1
            for prc, amt in sorted(orderdepth.sell_orders.items(), reverse=False):
                cls._hist_order_depths[product][f"askamt{cnt}"].append(amt)
                cls._hist_order_depths[product][f"askprc{cnt}"].append(prc)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f"askprc{cnt}"].append(np.nan)
                cls._hist_order_depths[product][f"askamt{cnt}"].append(np.nan)
                cnt += 1
            cnt = 1
            for prc, amt in sorted(orderdepth.buy_orders.items(), reverse=True):
                cls._hist_order_depths[product][f"bidprc{cnt}"].append(prc)
                cls._hist_order_depths[product][f"bidamt{cnt}"].append(amt)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f"bidprc{cnt}"].append(np.nan)
                cls._hist_order_depths[product][f"bidamt{cnt}"].append(np.nan)
                cnt += 1
        cls._num_data += 1

    def hist_order_depth(self, type: str, depth: int, size) -> np.ndarray:
        """Return historical order depth.

        Args:
            type (str): 'bidprc' or 'bidamt' or 'askprc' or 'askamt'
            depth (int): depth, 1 or 2 or 3
            size (int): size of data

        Returns:
            np.ndarray: historical order depth for given type and depth

        """
        return np.array(
            self._hist_order_depths[self.product][f"{type}{depth}"][-size:],
            dtype=np.float32,
        )

    @property
    def timestep(self) -> int:
        return self._state.timestamp / 100

    @property
    def position_limit(self) -> int:
        """Return position limit of product.

        Returns:
            int: position limit of product

        """
        return self._position_limit[self.product]

    @property
    def position(self) -> int:
        """Return current position of product.

        Returns:
            int: current position of product

        """
        if self.product in self._state.position:
            return int(self._state.position[self.product])
        else:
            return 0

    @property
    def rt_position(self) -> int:
        """Return realtime position.

        Returns:
            int: realtime position

        """
        return self._realtime_position[self.product]

    def _cls_rt_position_update(cls, product, new_position):
        if abs(new_position) <= cls._position_limit[product]:
            cls._realtime_position[product] = new_position
        else:
            raise ValueError("New position exceeds position limit")

    def rt_position_update(self, new_position: int) -> None:
        """Update realtime position.

        Args:
            new_position (int): new position

        """
        self._cls_rt_position_update(self.product, new_position)

    @property
    def bids(self) -> list[tuple[int, int]]:
        """Return bid orders.

        Returns:
            dict[int, int].items(): bid orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].buy_orders.items())

    @property
    def asks(self) -> list[tuple[int, int]]:
        """Return ask orders.

        Returns:
            dict[int, int].items(): ask orders (prc, amt)

        """
        return list(self._state.order_depths[self.product].sell_orders.items())

    @classmethod
    def _cls_update_bids(cls, product, prc, new_amt):
        if new_amt > 0:
            cls._state.order_depths[product].buy_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].buy_orders[prc] = 0

    @classmethod
    def _cls_update_asks(cls, product, prc, new_amt):
        if new_amt < 0:
            cls._state.order_depths[product].sell_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].sell_orders[prc] = 0

    def update_bids(self, prc: int, new_amt: int) -> None:
        """Update bid orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_bids(self.product, prc, new_amt)

    def update_asks(self, prc: int, new_amt: int) -> None:
        """Update ask orders.

        Args:
            prc (int): price
            new_amt (int): new amount

        """
        self._cls_update_asks(self.product, prc, new_amt)

    @property
    def possible_buy_amt(self) -> int:
        """Return possible buy amount.

        Returns:
            int: possible buy amount

        """
        possible_buy_amount1 = self._position_limit[self.product] - self.rt_position
        possible_buy_amount2 = self._position_limit[self.product] - self.position
        return min(possible_buy_amount1, possible_buy_amount2)

    @property
    def possible_sell_amt(self) -> int:
        """Return possible sell amount.

        Returns:
            int: possible sell amount

        """
        possible_sell_amount1 = self._position_limit[self.product] + self.rt_position
        possible_sell_amount2 = self._position_limit[self.product] + self.position
        return min(possible_sell_amount1, possible_sell_amount2)

    def hist_mid_prc(self, size: int) -> np.ndarray:
        """Return historical mid price.

        Args:
            size (int): size of data

        Returns:
            np.ndarray: historical mid price

        """
        return (
            self.hist_order_depth("bidprc", 1, size)
            + self.hist_order_depth("askprc", 1, size)
        ) / 2

    def hist_maxamt_askprc(self, size: int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth

        """
        res_array = np.empty(size)
        prc_array = np.array(
            [
                self.hist_order_depth("askprc", 1, size),
                self.hist_order_depth("askprc", 2, size),
                self.hist_order_depth("askprc", 3, size),
            ]
        ).T
        amt_array = np.array(
            [
                self.hist_order_depth("askamt", 1, size),
                self.hist_order_depth("askamt", 2, size),
                self.hist_order_depth("askamt", 3, size),
            ]
        ).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i, np.nanargmax(amt_arr)]

        return res_array

    def hist_maxamt_bidprc(self, size: int) -> np.ndarray:
        """Return price of ask order with maximum amount in historical order depth.

        Args:
            size (int): size of data

        Returns:
            int: price of ask order with maximum amount in historical order depth

        """
        res_array = np.empty(size)
        prc_array = np.array(
            [
                self.hist_order_depth("bidprc", 1, size),
                self.hist_order_depth("bidprc", 2, size),
                self.hist_order_depth("bidprc", 3, size),
            ]
        ).T
        amt_array = np.array(
            [
                self.hist_order_depth("bidamt", 1, size),
                self.hist_order_depth("bidamt", 2, size),
                self.hist_order_depth("bidamt", 3, size),
            ]
        ).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i, np.nanargmax(amt_arr)]

        return res_array

    def hist_vwap_all(self, size: int) -> np.ndarray:
        res_array = np.zeros(size)
        volsum_array = np.zeros(size)
        for i in range(1, 4):
            tmp_bid_vol = self.hist_order_depth(f"bidamt", i, size)
            tmp_ask_vol = self.hist_order_depth(f"askamt", i, size)
            tmp_bid_prc = self.hist_order_depth(f"bidprc", i, size)
            tmp_ask_prc = self.hist_order_depth(f"askprc", i, size)
            if i == 0:
                res_array = (
                    res_array
                    + (tmp_bid_prc * tmp_bid_vol)
                    + (-tmp_ask_prc * tmp_ask_vol)
                )
                volsum_array = volsum_array + tmp_bid_vol - tmp_ask_vol
            else:
                bid_nan_idx = np.isnan(tmp_bid_prc)
                ask_nan_idx = np.isnan(tmp_ask_prc)
                res_array = (
                    res_array
                    + np.where(bid_nan_idx, 0, tmp_bid_prc * tmp_bid_vol)
                    + np.where(ask_nan_idx, 0, -tmp_ask_prc * tmp_ask_vol)
                )
                volsum_array = (
                    volsum_array
                    + np.where(bid_nan_idx, 0, tmp_bid_vol)
                    - np.where(ask_nan_idx, 0, tmp_ask_vol)
                )

        return res_array / volsum_array

    def hist_obs_humidity(self, size: int) -> np.ndarray:
        return np.array(self._hist_observation["humidity"][-size:], dtype=np.float32)

    def hist_obs_sunlight(self, size: int) -> np.ndarray:
        return np.array(self._hist_observation["sunlight"][-size:], dtype=np.float32)

    def hist_obs_transportFees(self, size: int) -> np.ndarray:
        return np.array(
            self._hist_observation["transportFees"][-size:], dtype=np.float32
        )

    def hist_obs_exportTariff(self, size: int) -> np.ndarray:
        return np.array(
            self._hist_observation["exportTariff"][-size:], dtype=np.float32
        )

    def hist_obs_importTariff(self, size: int) -> np.ndarray:
        return np.array(
            self._hist_observation["importTariff"][-size:], dtype=np.float32
        )

    def hist_obs_bidPrice(self, size: int) -> np.ndarray:
        return np.array(self._hist_observation["bidPrice"][-size:], dtype=np.float32)

    def hist_obs_askPrice(self, size: int) -> np.ndarray:
        return np.array(self._hist_observation["askPrice"][-size:], dtype=np.float32)

    @property
    def best_bid(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)

        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return max(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def best_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return min(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def bid_ask_spread(self) -> int:
        return self.best_ask - self.best_bid

    @property
    def best_bid_amount(self) -> int:
        """Return best bid price and amount.

        Returns:
            tuple[int, int]: (price, amount)

        """
        best_prc = max(self._state.order_depths[self.product].buy_orders.keys())
        best_amt = self._state.order_depths[self.product].buy_orders[best_prc]
        return best_amt

    @property
    def best_ask_amount(self) -> int:
        """Return best ask price and amount.

        Returns:
            tuple[int, int]: (price, amount)

        """
        best_prc = min(self._state.order_depths[self.product].sell_orders.keys())
        best_amt = self._state.order_depths[self.product].sell_orders[best_prc]
        return -best_amt

    @property
    def worst_bid(self) -> int:
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return min(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def worst_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return max(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def vwap(self) -> float:
        vwap = 0
        total_amt = 0

        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += prc * amt
            total_amt += amt

        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += prc * abs(amt)
            total_amt += abs(amt)

        vwap /= total_amt
        return vwap

    @property
    def vwap_bidprc(self) -> float:
        """Return volume weighted average price of bid orders.

        Returns:
            float: volume weighted average price of bid orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += prc * amt
        vwap /= sum(self._state.order_depths[self.product].buy_orders.values())
        return vwap

    @property
    def vwap_askprc(self) -> float:
        """Return volume weighted average price of ask orders.

        Returns:
            float: volume weighted average price of ask orders

        """
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += prc * -amt
        vwap /= -sum(self._state.order_depths[self.product].sell_orders.values())
        return vwap

    @property
    def maxamt_bidprc(self) -> int:
        """Return price of bid order with maximum amount.

        Returns:
            int: price of bid order with maximum amount

        """
        prc_max_mat, max_amt = 0, 0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            if amt > max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat

    @property
    def maxamt_askprc(self) -> int:
        """Return price of ask order with maximum amount.

        Returns:
            int: price of ask order with maximum amount

        """
        prc_max_mat, max_amt = 0, 0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            if amt < max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat

    @property
    def maxamt_midprc(self) -> float:
        return (self.maxamt_bidprc + self.maxamt_askprc) / 2

    def bidamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].buy_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0

    def askamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].sell_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0

    @property
    def total_bidamt(self) -> int:
        return sum(self._state.order_depths[self.product].buy_orders.values())

    @property
    def total_askamt(self) -> int:
        return -sum(self._state.order_depths[self.product].sell_orders.values())

    @property
    def orchid_south_bidprc(self) -> float:
        return self._state.observations.conversionObservations[self.product].bidPrice

    @property
    def orchid_south_askprc(self) -> float:
        return self._state.observations.conversionObservations[self.product].askPrice

    @property
    def orchid_south_midprc(self) -> float:
        return (self.orchid_south_bidprc + self.orchid_south_askprc) / 2

    @property
    def stoarageFees(self) -> float:
        return 0.1

    @property
    def transportFees(self) -> float:
        return self._state.observations.conversionObservations[
            self.product
        ].transportFees

    @property
    def exportTariff(self) -> float:
        return self._state.observations.conversionObservations[
            self.product
        ].exportTariff

    @property
    def importTariff(self) -> float:
        return self._state.observations.conversionObservations[
            self.product
        ].importTariff

    @property
    def sunlight(self) -> float:
        return self._state.observations.conversionObservations[self.product].sunlight

    @property
    def humidity(self) -> float:
        return self._state.observations.conversionObservations[self.product].humidity

    @property
    def market_trades(self) -> list:
        return self._state.market_trades.get(self.product, [])


def linear_regression(X, y):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)
    return theta


def cal_tau(day, timestep, T=1):
    return T - ((day - 1) * 20000 + timestep) * 2e-7


def cal_call(S, tau, sigma=0.16, r=0, K=10000):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))
    delta = normalDist.cdf(d1)
    d2 = d1 - sigma * np.sqrt(tau)
    call_price = S * delta - K * math.exp(-r * tau) * normalDist.cdf(d2)
    return call_price, delta


def cal_imvol(market_price, S, tau, r=0, K=10000, tol=1e-6, max_iter=100):
    sigma = 0.16
    diff = cal_call(S, tau, sigma)[0] - market_price

    iter_count = 0
    while np.any(np.abs(diff) > tol) and iter_count < max_iter:
        vega = (cal_call(S, tau, sigma + tol)[0] - cal_call(S, tau, sigma)[0]) / tol
        sigma -= diff / vega
        diff = cal_call(S, tau, sigma)[0] - market_price
        iter_count += 1

    return sigma


def cal_obi(bid_vol: float, ask_vol: float) -> float:
    if bid_vol is None or ask_vol is None or (bid_vol + ask_vol == 0):
        return 0.0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)


def calc_kelp_next_price(kelp_mid_price_history):
    coef = [0.32744102, 0.2609162, 0.20492715, 0.20581442]
    intercept = 1.8260964
    nxt_price = intercept
    for i, val in enumerate(kelp_mid_price_history):
        nxt_price += val * coef[i]
    return int(round(nxt_price))


def ewma(ls, alpha=0.6):
    final_res = ls[0]
    for i in range(1, len(ls)):
        final_res = alpha * ls[i] + (1 - alpha) * final_res
    return final_res


class SignalType(Enum):
    BUY = 1
    SELL = 2
    NOTHING = 3
    UNLOAD = 4


def gen_signal(z_score, z_threshold, ema_diff, ema_edge_open, ema_edge_close, cur_pos):
    # z_score as reversion signal
    if abs(z_score) > z_threshold:
        logger.print("reversion signal")
        direction = -1 if z_score > 0 else 1
        if direction == 1:
            return SignalType.BUY
        elif direction == -1:
            return SignalType.SELL

    # ema_diff as trend signal
    if cur_pos == 0:
        print("trend signal")
        if ema_diff > ema_edge_open:
            return SignalType.BUY
        elif ema_diff < -ema_edge_open:
            return SignalType.SELL
        return SignalType.NOTHING
    elif cur_pos > 0:
        if ema_diff < -ema_edge_close:
            return SignalType.SELL
        return SignalType.BUY if ema_diff > ema_edge_open else SignalType.NOTHING
    elif cur_pos < 0:
        if ema_diff > ema_edge_close:
            return SignalType.BUY
        return SignalType.SELL if ema_diff < -ema_edge_open else SignalType.NOTHING

    return SignalType.NOTHING


def gen_order_size(cur_pos, limit_pos, this_signal: SignalType):
    if this_signal == SignalType.BUY:
        cur_pctg = cur_pos / limit_pos
        if cur_pctg > 0.7:
            return limit_pos - cur_pos
        if cur_pctg > 0.5:
            return int(0.3 * limit_pos)
        return int(0.5 * limit_pos)
    elif this_signal == SignalType.SELL:
        cur_pctg = -cur_pos / limit_pos
        if cur_pctg > 0.7:
            return -(limit_pos + cur_pos)
        if cur_pctg > 0.5:
            return int(-0.3 * limit_pos)
        return int(-0.5 * limit_pos)
    elif this_signal == SignalType.UNLOAD:
        return -cur_pos


class ExecutionProb:
    @staticmethod
    def orchids(delta):
        if delta < -1:
            return 0.571
        elif delta > -0.5:
            return 0
        elif delta == -1.0:
            return 0.2685
        elif delta == -0.75:
            return 0.2107
        elif delta == -0.5:
            return 0.1699


class Strategy:
    slope_history = deque(maxlen=50)

    @staticmethod
    def arb(state: Status, fair_price):
        orders = []

        for ask_price, ask_amount in state.asks:
            if ask_price < fair_price:
                buy_amount = min(-ask_amount, state.possible_buy_amt)
                if buy_amount > 0:
                    orders.append(Order(state.product, int(ask_price), int(buy_amount)))
                    state.rt_position_update(state.rt_position + buy_amount)
                    state.update_asks(ask_price, -(-ask_amount - buy_amount))

            elif ask_price == fair_price:
                if state.rt_position < 0:
                    buy_amount = min(-ask_amount, -state.rt_position)
                    orders.append(Order(state.product, int(ask_price), int(buy_amount)))
                    state.rt_position_update(state.rt_position + buy_amount)
                    state.update_asks(ask_price, -(-ask_amount - buy_amount))

        for bid_price, bid_amount in state.bids:
            if bid_price > fair_price:
                sell_amount = min(bid_amount, state.possible_sell_amt)
                if sell_amount > 0:
                    orders.append(
                        Order(state.product, int(bid_price), -int(sell_amount))
                    )
                    state.rt_position_update(state.rt_position - sell_amount)
                    state.update_bids(bid_price, bid_amount - sell_amount)

            elif bid_price == fair_price:
                if state.rt_position > 0:
                    sell_amount = min(bid_amount, state.rt_position)
                    orders.append(
                        Order(state.product, int(bid_price), -int(sell_amount))
                    )
                    state.rt_position_update(state.rt_position - sell_amount)
                    state.update_bids(bid_price, bid_amount - sell_amount)

        return orders

    @staticmethod
    def mm_glft(
        state: Status,
        fair_price,
        mu=0,
        sigma=0.3959,
        gamma=1e-9,
        order_amount=20,
    ):

        q = state.rt_position / order_amount
        # Q = state.position_limit / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)

        A_b = 0.25
        A_a = 0.25

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) + (
            -mu / (gamma * sigma**2) + (2 * q + 1) / 2
        ) * math.sqrt(
            (sigma**2 * gamma)
            / (2 * kappa_b * A_b)
            * (1 + gamma / kappa_b) ** (1 + kappa_b / gamma)
        )
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + (
            mu / (gamma * sigma**2) - (2 * q - 1) / 2
        ) * math.sqrt(
            (sigma**2 * gamma)
            / (2 * kappa_a * A_a)
            * (1 + gamma / kappa_a) ** (1 + kappa_a / gamma)
        )

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(
            p_b, fair_price
        )  # Set the buy price to be no higher than the fair price to avoid losses
        p_b = min(
            p_b, state.best_bid + 1
        )  # Place the buy order as close as possible to the best bid price
        p_b = max(
            p_b, state.maxamt_bidprc + 1
        )  # No market order arrival beyond this price

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        orders = []
        if buy_amount > 0:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders

    def mm_glft_slope(
        state: Status,
        fair_price,
        mu=0,
        sigma=0.3959,
        gamma=1e-9,
        order_amount=20,
        avg_slope=0,
        slope_thresh=0.02,
    ):

        q = state.rt_position / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)

        A_b = 0.25
        A_a = 0.25

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) + (
            -mu / (gamma * sigma**2) + (2 * q + 1) / 2
        ) * math.sqrt(
            (sigma**2 * gamma)
            / (2 * kappa_b * A_b)
            * (1 + gamma / kappa_b) ** (1 + kappa_b / gamma)
        )

        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + (
            mu / (gamma * sigma**2) - (2 * q - 1) / 2
        ) * math.sqrt(
            (sigma**2 * gamma)
            / (2 * kappa_a * A_a)
            * (1 + gamma / kappa_a) ** (1 + kappa_a / gamma)
        )

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(p_b, fair_price)
        p_b = min(p_b, state.best_bid + 1)
        p_b = max(p_b, state.maxamt_bidprc + 1)

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        skip_buy = avg_slope < -slope_thresh and state.rt_position > 0
        skip_sell = avg_slope > slope_thresh and state.rt_position < 0

        orders = []
        if buy_amount > 0 and not skip_buy:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0 and not skip_sell:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders

    @staticmethod
    def mm_ou(
        state: Status,
        fair_price,
        gamma=1e-9,
        order_amount=20,
    ):

        q = state.rt_position / order_amount
        Q = state.position_limit / order_amount

        kappa_b = 1 / max((fair_price - state.best_bid) - 1, 1)
        kappa_a = 1 / max((state.best_ask - fair_price) - 1, 1)

        vfucn = lambda q, Q: (
            -INF
            if (q == Q + 1 or q == -(Q + 1))
            else math.log(math.sin(((q + Q + 1) * math.pi) / (2 * Q + 2)))
        )

        delta_b = 1 / gamma * math.log(1 + gamma / kappa_b) - 1 / kappa_b * (
            vfucn(q + 1, Q) - vfucn(q, Q)
        )
        delta_a = 1 / gamma * math.log(1 + gamma / kappa_a) + 1 / kappa_a * (
            vfucn(q, Q) - vfucn(q - 1, Q)
        )

        p_b = round(fair_price - delta_b)
        p_a = round(fair_price + delta_a)

        p_b = min(
            p_b, fair_price
        )  # Set the buy price to be no higher than the fair price to avoid losses
        p_b = min(
            p_b, state.best_bid + 1
        )  # Place the buy order as close as possible to the best bid price
        p_b = max(
            p_b, state.maxamt_bidprc + 1
        )  # No market order arrival beyond this price

        p_a = max(p_a, fair_price)
        p_a = max(p_a, state.best_ask - 1)
        p_a = min(p_a, state.maxamt_askprc - 1)

        buy_amount = min(order_amount, state.possible_buy_amt)
        sell_amount = min(order_amount, state.possible_sell_amt)

        orders = []
        if buy_amount > 0:
            orders.append(Order(state.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(state.product, int(p_a), -int(sell_amount)))
        return orders

    @staticmethod
    def exchange_arb(state: Status, fair_price, next_price_move=0):
        cost = state.transportFees + state.importTariff

        my_ask = state.maxamt_bidprc
        ask_max_expected_profit = 0
        optimal_my_ask = INF
        while my_ask < fair_price:
            delta = my_ask - fair_price
            execution_prob = ExecutionProb.orchids(delta)

            if my_ask > state.best_bid:
                trading_profit = my_ask - (state.orchid_south_askprc + next_price_move)
                expected_profit = execution_prob * (trading_profit - cost)

            else:
                execution_prob_list = []
                price_list = []
                amount_list = []

                for price, amount in state.bids:
                    if price >= my_ask:
                        execution_prob_list.append(1)
                        price_list.append(price)
                        amount_list.append(amount)

                total_amount = np.sum(amount_list)
                if total_amount < state.position_limit:
                    execution_prob_list.append(ExecutionProb.orchids(delta))
                    price_list.append(my_ask)
                    amount_list.append(state.position_limit - total_amount)

                trading_profit_list = np.array(price_list) - (
                    state.orchid_south_askprc + next_price_move
                )
                expected_profit = (
                    np.array(execution_prob_list)
                    * (np.array(trading_profit_list) - cost)
                    * np.array(amount_list)
                    / state.position_limit
                ).sum()

            if expected_profit > ask_max_expected_profit:
                optimal_my_ask = my_ask
                ask_max_expected_profit = expected_profit

            my_ask += 1

        cost = state.transportFees + state.exportTariff + state.stoarageFees

        my_bid = state.maxamt_askprc
        bid_max_expected_profit = 0
        optimal_my_bid = 1
        while my_bid > fair_price:
            delta = fair_price - my_bid
            execution_prob = ExecutionProb.orchids(delta)

            if my_bid < state.best_ask:
                trading_profit = (state.orchid_south_bidprc + next_price_move) - my_bid
                expected_profit = execution_prob * (trading_profit - cost)

            else:
                execution_prob_list = []
                price_list = []
                amount_list = []

                for price, amount in state.asks:
                    if price <= my_bid:
                        execution_prob_list.append(1)
                        price_list.append(price)
                        amount_list.append(abs(amount))

                total_amount = np.sum(amount_list)
                if total_amount < state.position_limit:
                    execution_prob_list.append(ExecutionProb.orchids(delta))
                    price_list.append(my_bid)
                    amount_list.append(state.position_limit - total_amount)

                trading_profit_list = (
                    state.orchid_south_bidprc + next_price_move
                ) - np.array(price_list)
                expected_profit = (
                    np.array(execution_prob_list)
                    * (np.array(trading_profit_list) - cost)
                    * np.array(amount_list)
                    / state.position_limit
                ).sum()

            if expected_profit > bid_max_expected_profit:
                optimal_my_bid = my_bid
                bid_max_expected_profit = expected_profit

            my_bid -= 1

        orders = []

        if (
            ask_max_expected_profit >= bid_max_expected_profit
            and ask_max_expected_profit > 0
        ):
            orders.append(
                Order(state.product, int(optimal_my_ask), -int(state.position_limit))
            )
        elif (
            bid_max_expected_profit > ask_max_expected_profit
            and bid_max_expected_profit > 0
        ):
            orders.append(
                Order(state.product, int(optimal_my_bid), int(state.position_limit))
            )

        return orders

    @staticmethod
    def convert(state: Status):
        if state.position < 0:
            return -state.position
        elif state.position > 0:
            return -state.position
        else:
            return 0

    @staticmethod
    def index_arb(
        basket: Status,
        chocolate: Status,
        strawberries: Status,
        roses: Status,
        theta=380,
        threshold=30,
    ):

        basket_prc = basket.mid
        underlying_prc = 4 * chocolate.vwap + 6 * strawberries.vwap + 1 * roses.vwap
        spread = basket_prc - underlying_prc
        norm_spread = spread - theta

        orders = []
        if norm_spread > threshold:
            orders.append(
                Order(
                    basket.product,
                    int(basket.worst_bid),
                    -int(basket.possible_sell_amt),
                )
            )
        elif norm_spread < -threshold:
            orders.append(
                Order(
                    basket.product, int(basket.worst_ask), int(basket.possible_buy_amt)
                )
            )

        return orders

    @staticmethod
    def index_arb1(
        basket: Status,
        croissant: Status,
        jam: Status,
        djembe: Status,
        theta=380,
        open_threshold=30,
        close_threshold=10,
    ):
        """
        arb for picnic_basket1
        """
        real_best_bid = basket.best_bid
        real_best_ask = basket.best_ask
        etf_best_bid = 6 * croissant.best_bid + 3 * jam.best_bid + 1 * djembe.best_bid
        etf_best_ask = 6 * croissant.best_ask + 3 * jam.best_ask + 1 * djembe.best_ask
        buy_real_price_diff = max(etf_best_bid - real_best_ask, 0)
        sell_real_price_diff = max(real_best_bid - etf_best_ask, 0)
        cur_arb_pos = basket.rt_position
        orders = []

        if cur_arb_pos > 0 and buy_real_price_diff < close_threshold:
            # open position
            possible_sell_amt = min(
                basket.best_bid_amount,
                croissant.best_ask_amount // 6,
                jam.best_ask_amount // 3,
                djembe.best_ask_amount,
                cur_arb_pos,
            )
            orders.append(
                Order(
                    basket.product,
                    int(real_best_bid),
                    -int(possible_sell_amt),
                )
            )
            orders.append(
                Order(
                    croissant.product,
                    int(croissant.best_ask),
                    int(possible_sell_amt * 6),
                )
            )
            orders.append(
                Order(
                    jam.product,
                    int(jam.best_ask),
                    int(possible_sell_amt * 3),
                )
            )
            orders.append(
                Order(
                    djembe.product,
                    int(djembe.best_ask),
                    int(possible_sell_amt),
                )
            )
            cur_arb_pos -= possible_sell_amt
            croissant.rt_position_update(
                croissant.rt_position + int(possible_sell_amt * 6)
            )
            jam.rt_position_update(jam.rt_position + int(possible_sell_amt * 3))
            djembe.rt_position_update(djembe.rt_position + int(possible_sell_amt))
        elif cur_arb_pos < 0 and sell_real_price_diff < close_threshold:
            possible_buy_amt = min(
                basket.best_ask_amount,
                croissant.best_bid_amount // 6,
                jam.best_bid_amount // 3,
                djembe.best_bid_amount,
                -cur_arb_pos,
            )
            orders.append(
                Order(
                    basket.product,
                    int(real_best_ask),
                    int(possible_buy_amt),
                )
            )
            orders.append(
                Order(
                    croissant.product,
                    int(croissant.best_bid),
                    -int(possible_buy_amt * 6),
                )
            )
            orders.append(
                Order(
                    jam.product,
                    int(jam.best_bid),
                    -int(possible_buy_amt * 3),
                )
            )
            orders.append(
                Order(
                    djembe.product,
                    int(djembe.best_bid),
                    -int(possible_buy_amt),
                )
            )
            cur_arb_pos += possible_buy_amt
            croissant.rt_position_update(
                croissant.rt_position - int(possible_buy_amt * 6)
            )
            jam.rt_position_update(jam.rt_position - int(possible_buy_amt * 3))
            djembe.rt_position_update(djembe.rt_position - int(possible_buy_amt))

        if buy_real_price_diff > open_threshold:
            possible_buy_amt = min(
                basket.best_ask_amount,
                croissant.best_bid_amount // 6,
                jam.best_bid_amount // 3,
                djembe.best_bid_amount,
                (basket.position_limit - basket.rt_position),
                (croissant.position_limit + croissant.rt_position) // 6,
                (jam.position_limit + jam.rt_position) // 3,
                (djembe.position_limit + djembe.rt_position),
            )
            orders.append(
                Order(
                    basket.product,
                    int(real_best_ask),
                    int(possible_buy_amt),
                )
            )
            orders.append(
                Order(
                    croissant.product,
                    int(croissant.best_bid),
                    -int(possible_buy_amt * 6),
                )
            )
            orders.append(
                Order(
                    jam.product,
                    int(jam.best_bid),
                    -int(possible_buy_amt * 3),
                )
            )
            orders.append(
                Order(
                    djembe.product,
                    int(djembe.best_bid),
                    -int(possible_buy_amt),
                )
            )
        elif sell_real_price_diff > open_threshold:
            possible_sell_amt = min(
                basket.best_bid_amount,
                croissant.best_ask_amount // 6,
                jam.best_ask_amount // 3,
                djembe.best_ask_amount,
                (basket.position_limit + basket.rt_position),
                (croissant.position_limit - croissant.rt_position) // 6,
                (jam.position_limit - jam.rt_position) // 3,
                (djembe.position_limit - djembe.rt_position),
            )
            orders.append(
                Order(
                    basket.product,
                    int(real_best_bid),
                    -int(possible_sell_amt),
                )
            )
            orders.append(
                Order(
                    croissant.product,
                    int(croissant.best_ask),
                    int(possible_sell_amt * 6),
                )
            )
            orders.append(
                Order(
                    jam.product,
                    int(jam.best_ask),
                    int(possible_sell_amt * 3),
                )
            )
            orders.append(
                Order(
                    djembe.product,
                    int(djembe.best_ask),
                    int(possible_sell_amt),
                )
            )

        return orders

    @staticmethod
    def vol_arb(option: Status, iv, hv=0.16, threshold=0.00178):

        vol_spread = iv - hv

        orders = []

        if vol_spread > threshold:
            sell_amount = option.possible_sell_amt
            orders.append(Order(option.product, option.worst_bid, -sell_amount))
            executed_amount = min(sell_amount, option.total_bidamt)
            option.rt_position_update(option.rt_position - executed_amount)

        elif vol_spread < -threshold:
            buy_amount = option.possible_buy_amt
            orders.append(Order(option.product, option.worst_ask, buy_amount))
            executed_amount = min(buy_amount, option.total_askamt)
            option.rt_position_update(option.rt_position + executed_amount)

        return orders

    @staticmethod
    def delta_hedge(underlying: Status, option: Status, delta, rebalance_threshold=30):

        target_position = -round(option.rt_position * delta)
        current_position = underlying.position
        position_diff = target_position - current_position

        orders = []

        if underlying.bid_ask_spread == 1 and abs(position_diff) > rebalance_threshold:

            if position_diff < 0:
                sell_amount = min(abs(position_diff), underlying.possible_sell_amt)
                orders.append(
                    Order(underlying.product, underlying.best_bid, -sell_amount)
                )

            elif position_diff > 0:
                buy_amount = min(position_diff, underlying.possible_buy_amt)
                orders.append(
                    Order(underlying.product, underlying.best_ask, buy_amount)
                )

        return orders

    @staticmethod
    def insider_trading(signal_product: Status, trade_product: Status):

        buy_timestamp, sell_timestamp = 0, 0

        for trade in signal_product.market_trades:
            if trade.buyer == "Rhianna":
                buy_timestamp = trade.timestamp
            elif trade.seller == "Rhianna":
                sell_timestamp = trade.timestamp

        orders = []
        if buy_timestamp > sell_timestamp:
            orders.append(
                Order(
                    trade_product.product,
                    trade_product.worst_ask,
                    trade_product.possible_buy_amt,
                )
            )
        elif buy_timestamp < sell_timestamp:
            orders.append(
                Order(
                    trade_product.product,
                    trade_product.worst_bid,
                    -trade_product.possible_sell_amt,
                )
            )

        return orders

    @staticmethod
    def market_edge_orders(product: Status, acc_bid: int, acc_ask: int) -> list[Order]:
        orders = []
        best_sell_pr = product.best_ask
        best_buy_pr = product.best_bid
        cur_pos = product.rt_position
        pos_limit = product.position_limit

        for ask, vol in sorted(product.asks):
            if (ask <= acc_bid and cur_pos < pos_limit) or (
                cur_pos < 0 and ask == acc_bid + 1
            ):
                buy_amt = min(-vol, pos_limit - cur_pos)
                cur_pos += buy_amt
                assert buy_amt >= 0
                orders.append(Order(product.product, ask, buy_amt))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)
        sell_pr = max(undercut_sell, acc_ask)

        if cur_pos < pos_limit:
            num = pos_limit - cur_pos
            orders.append(Order(product.product, bid_pr, num))
            cur_pos += num

        cur_pos = product.rt_position

        for bid, vol in sorted(product.bids, reverse=True):
            if (bid >= acc_ask and cur_pos > -pos_limit) or (
                cur_pos > 0 and bid + 1 == acc_ask
            ):
                sell_amt = max(-vol, -pos_limit - cur_pos)
                cur_pos += sell_amt
                assert sell_amt <= 0
                orders.append(Order(product.product, bid, sell_amt))

        if cur_pos > -pos_limit:
            num = -cur_pos - pos_limit
            orders.append(Order(product.product, sell_pr, num))
            cur_pos += num
        return orders

    @staticmethod
    def smart_place_order(
        product: Status, order_amt: int, is_exit: bool
    ) -> list[Order]:
        orders = []
        if order_amt > 0:
            # buy
            if is_exit:
                orders.append(
                    Order(
                        product.product,
                        min(product.best_bid + 2, product.best_ask),
                        order_amt,
                    )
                )
            else:
                orders.append(Order(product.product, product.best_bid, order_amt))
        elif order_amt < 0:
            # sell
            if is_exit:
                orders.append(
                    Order(
                        product.product,
                        max(product.best_bid, product.best_ask - 2),
                        order_amt,
                    )
                )
            else:
                orders.append(Order(product.product, product.best_ask, order_amt))
        return orders

    @staticmethod
    def take_best_orders(
        product: Status,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[int, int]:

        position_limit = product.position_limit

        if product.best_ask_amount > 0:
            best_ask = product.best_ask
            best_ask_amount = product.best_ask_amount
            if prevent_adverse:
                if (
                    best_ask_amount <= adverse_volume
                    and best_ask <= fair_value - take_width
                ):
                    quantity = min(
                        best_ask_amount, position_limit - product.rt_position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product.product, best_ask, quantity))
                        buy_order_volume += quantity
            else:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - product.rt_position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product.product, best_ask, quantity))
                        buy_order_volume += quantity

        if product.best_bid_amount > 0:
            best_bid = product.best_bid
            best_bid_amount = product.best_bid_amount
            if prevent_adverse:
                if (best_bid >= fair_value + take_width) and (
                    best_bid_amount <= adverse_volume
                ):
                    quantity = min(
                        best_bid_amount, position_limit + product.rt_position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product.product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
            else:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + product.rt_position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product.product, best_bid, -1 * quantity))
                        sell_order_volume += quantity

        return buy_order_volume, sell_order_volume

    @staticmethod
    def clear_position_order(
        product: Status,
        fair_value: float,
        width: int,
        orders: List[Order],
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:

        position_after_take = product.rt_position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        buy_quantity = product.position_limit - (product.rt_position + buy_order_volume)
        sell_quantity = product.position_limit + (
            product.rt_position - sell_order_volume
        )

        if position_after_take > 0:
            bid_prices = [bid[0] for bid in product.bids]
            if fair_for_ask in bid_prices:
                clear_quantity = min(product.bidamt(fair_for_ask), position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product.product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            ask_prices = [ask[0] for ask in product.asks]
            if fair_for_bid in ask_prices:
                clear_quantity = min(
                    abs(product.askamt(fair_for_bid)), abs(position_after_take)
                )
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product.product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    @staticmethod
    def market_make(
        product: Status,
        orders: List[Order],
        bid: int,
        ask: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:

        buy_quantity = product.position_limit - (product.rt_position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product.product, bid, buy_quantity))  # Buy order

        sell_quantity = product.position_limit + (
            product.rt_position - sell_order_volume
        )
        if sell_quantity > 0:
            orders.append(Order(product.product, ask, -sell_quantity))  # Sell order

        return buy_order_volume, sell_order_volume

    @staticmethod
    def calc_fair_value(product: Status, method="mid_price", min_vol=0) -> float:
        if method == "mid_price":
            best_ask = product.best_ask
            best_bid = product.best_bid
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            sell_prices = [x[0] for x in product.asks]
            buy_prices = [x[0] for x in product.bids]
            if (
                len(
                    [
                        price
                        for price in sell_prices
                        if abs(product.askamt(price)) >= min_vol
                    ]
                )
                == 0
                or len(
                    [
                        price
                        for price in buy_prices
                        if abs(product.bidamt(price)) >= min_vol
                    ]
                )
                == 0
            ):
                best_ask = product.best_ask
                best_bid = product.best_bid
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:
                best_ask = min(
                    [
                        price
                        for price in sell_prices
                        if abs(product.askamt(price)) >= min_vol
                    ]
                )
                best_bid = max(
                    [
                        price
                        for price in buy_prices
                        if abs(product.bidamt(price)) >= min_vol
                    ]
                )
                mid_price = (best_ask + best_bid) / 2
            return mid_price


class Trade:
    @staticmethod
    def resin(state: Status) -> list[Order]:
        """
        market making using mm_ou
        """
        current_price = state.vwap
        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=current_price))
        orders.extend(
            Strategy.mm_ou(
                state=state, fair_price=current_price, gamma=1e-7, order_amount=50
            )
        )
        return orders

    def resin_edgar(state: Status) -> list[Order]:
        """
        market making from edgar
        """
        fair_value = 10_000
        width = 2
        buy_order_volume = 0
        sell_order_volume = 0
        ask_prices = [x[0] for x in state.asks]
        bid_prices = [x[0] for x in state.bids]
        baaf = min([price for price in ask_prices if price > fair_value + 1])
        bbbf = max([price for price in bid_prices if price < fair_value - 1])
        orders = []
        buy_order_volume, sell_order_volume = Strategy.take_best_orders(
            state,
            fair_value,
            width,
            orders,
            buy_order_volume,
            sell_order_volume,
        )
        buy_order_volume, sell_order_volume = Strategy.clear_position_order(
            state,
            fair_value,
            width,
            orders,
            buy_order_volume,
            sell_order_volume,
        )
        buy_order_volume, sell_order_volume = Strategy.market_make(
            state,
            orders,
            bbbf + 1,
            baaf - 1,
            buy_order_volume,
            sell_order_volume,
        )
        return orders

    @staticmethod
    def kelp_lr_mm(
        state: Status, kelp_mid_price_history: deque, kelp_pred_window: int
    ) -> list[Order]:
        if len(kelp_mid_price_history) == kelp_pred_window:
            kelp_mid_price_history.popleft()
        kelp_mid_price_history.append(state.hist_mid_prc(1)[0])
        if len(kelp_mid_price_history) < kelp_pred_window:
            return []
        pred_fair_price = calc_kelp_next_price(kelp_mid_price_history)
        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=pred_fair_price))
        orders.extend(
            Strategy.mm_glft(
                state=state, fair_price=pred_fair_price, gamma=1e-5, order_amount=50
            )
        )
        return orders

    @staticmethod
    def kelp_lr_edge(
        state: Status, kelp_mid_price_history: deque, kelp_pred_window: int
    ) -> list[Order]:
        if len(kelp_mid_price_history) == kelp_pred_window:
            kelp_mid_price_history.popleft()
        kelp_mid_price_history.append(state.hist_mid_prc(1)[0])
        lb, ub = -1e9, 1e9
        if len(kelp_mid_price_history) == kelp_pred_window:
            next_kelp_price = calc_kelp_next_price(kelp_mid_price_history)
            lb, ub = next_kelp_price - 1, next_kelp_price + 1
        else:
            return []
        orders = Strategy.market_edge_orders(state, lb, ub)
        return orders

    @staticmethod
    def kelp_edgar(state: Status, kelp_prices: list, kelp_vwap: list) -> list[Order]:
        timespan = 10
        KELP_take_width = 1
        buy_order_volume = 0
        sell_order_volume = 0
        orders = []
        if len(state.asks) != 0 and len(state.bids) != 0:
            # Calculate Fair
            best_ask = state.best_ask
            best_bid = state.best_bid
            ask_prices = [x[0] for x in state.asks]
            bid_prices = [x[0] for x in state.bids]
            filtered_ask = [
                price for price in ask_prices if abs(state.askamt(price)) >= 15
            ]
            filtered_bid = [
                price for price in bid_prices if abs(state.bidamt(price)) >= 15
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            mmmid_price = (mm_ask + mm_bid) / 2
            kelp_prices.append(mmmid_price)

            # Calculate VWAP
            volume = state.best_ask_amount + state.best_bid_amount
            vwap = (
                best_bid * state.best_ask_amount + best_ask * state.best_bid_amount
            ) / volume
            kelp_vwap.append({"vol": volume, "vwap": vwap})

            if len(kelp_vwap) > timespan:
                kelp_vwap.pop(0)

            if len(kelp_prices) > timespan:
                kelp_prices.pop(0)

            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = Strategy.take_best_orders(
                state,
                fair_value,
                KELP_take_width,
                orders,
                buy_order_volume,
                sell_order_volume,
                True,
                20,
            )

            # Clear Position Orders
            buy_order_volume, sell_order_volume = Strategy.clear_position_order(
                state,
                fair_value,
                2,
                orders,
                buy_order_volume,
                sell_order_volume,
            )

            aaf = [price for price in ask_prices if price > fair_value + 1]
            bbf = [price for price in bid_prices if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = Strategy.market_make(
                state,
                orders,
                bbbf + 1,
                baaf - 1,
                buy_order_volume,
                sell_order_volume,
            )
        return orders

    @staticmethod
    def picnic_basket1(basket: Status, croissant: Status, jam: Status, djembe: Status):
        orders = []
        orders.extend(
            Strategy.index_arb1(
                basket,
                croissant,
                jam,
                djembe,
                open_threshold=100,
                close_threshold=20,
                theta=0,
            )
        )
        return orders

    @staticmethod
    def squid_ink_polyfit(state: Status) -> list[Order]:
        N = 20  # lookback window
        hist_mid = state.hist_mid_prc(N)

        try:
            if len(hist_mid) < N or state.best_bid is None or state.best_ask is None:
                fair_price = state.vwap
                x = np.arange(len(hist_mid))
                slope, intercept = np.polyfit(x, hist_mid, 1)
                Strategy.slope_history.append(slope)
                avg_slope = np.mean(Strategy.slope_history)

            else:
                x = np.arange(N)
                slope, intercept = np.polyfit(x, hist_mid, 1)
                Strategy.slope_history.append(slope)
                avg_slope = np.mean(Strategy.slope_history)

                bias = avg_slope * np.abs(avg_slope)

                bid_price = state.best_bid
                ask_price = state.best_ask
                bid_vol = state.best_bid_amount or 1
                ask_vol = state.best_ask_amount or 1

                ob_mid = (bid_price * ask_vol + ask_price * bid_vol) / (
                    bid_vol + ask_vol
                )

                trend_mid = hist_mid[-1] + bias
                fair_price = 0.5 * trend_mid + 0.5 * ob_mid

                padding = state.bid_ask_spread * 1.5
                fair_price = max(fair_price, bid_price - padding)
                fair_price = min(fair_price, ask_price + padding)
        except Exception as e:
            return []

        orders = []
        orders.extend(Strategy.arb(state=state, fair_price=fair_price))
        orders.extend(
            Strategy.mm_glft_slope(
                state=state,
                fair_price=fair_price,
                gamma=1e-4,
                order_amount=30,
                avg_slope=avg_slope,
                slope_thresh=0.01,
            )
        )

        return orders

    @staticmethod
    def squid_ink_momentum(state: Status) -> list[Order]:
        N = 20
        hist_mid = state.hist_mid_prc(N)

        if len(hist_mid) < N or state.best_bid is None or state.best_ask is None:
            return []

        returns = np.diff(hist_mid)
        long_term_mom = np.mean(returns[:-5])
        short_term_shock = np.mean(returns[-5:])
        threshold = 0.7 * np.std(returns)

        if long_term_mom > 0 and short_term_shock < -threshold:
            reversal_momentum = -abs(short_term_shock)
        elif long_term_mom < 0 and short_term_shock > threshold:
            reversal_momentum = abs(short_term_shock)
        else:
            reversal_momentum = 0

        bid_bias = (state.vwap - state.best_bid) / state.bid_ask_spread
        order_bias = 0.5 - bid_bias

        alpha = 2 / (N + 1)
        ema = hist_mid[0]
        for price in hist_mid[1:]:
            ema = alpha * price + (1 - alpha) * ema

        std = np.std(hist_mid)
        adj_vwap = (
            state.vwap
            + reversal_momentum * 0.4
            + order_bias * 0.6
            + long_term_mom * 0.2
        )
        lb = adj_vwap - 1 * std
        ub = adj_vwap + 1 * std
        return Strategy.market_edge_orders(state, int(round(lb)), int(round(ub)))

    @staticmethod
    def squid_ink_band(state: Status) -> list[Order]:
        N = 10
        hist_mid = state.hist_mid_prc(N)
        if len(hist_mid) < N or state.best_bid is None or state.best_ask is None:
            return []

        alpha = 2 / (N + 1)
        ema = hist_mid[0]
        for price in hist_mid[1:]:
            ema = alpha * price + (1 - alpha) * ema

        std = np.std(hist_mid)
        return Strategy.trend_follow_orders(state, ema, std)

    @staticmethod
    def squid_ink_da_arb(state: Status) -> list[Order]:
        N = 20
        if (
            len(state.hist_mid_prc(N)) < N
            or state.best_bid is None
            or state.best_ask is None
        ):
            return []
        Z_THRESHOLD = 1.5
        EMA_EDGE_OPEN = 1.2
        EMA_EDGE_CLOSE = 0.5
        hist_vwap = state.hist_vwap_all(N)

        # z_score as reversion signal
        if len(hist_vwap) < N:
            z_score = 0.0
        else:
            vwap_mean = np.mean(hist_vwap)
            vwap_std = np.std(hist_vwap)
            z_score = (np.mean(hist_vwap[-3:]) - vwap_mean) / (vwap_std + 1e-8)

        # ema_diff as trend signal
        if len(hist_vwap) < N:
            ema_diff = 0.0
        else:
            ema_long = ewma(hist_vwap, alpha=0.1)
            ema_short = ewma(hist_vwap, alpha=0.5)
            ema_diff = ema_short - ema_long

        this_signal = gen_signal(
            z_score=z_score,
            z_threshold=Z_THRESHOLD,
            ema_diff=ema_diff,
            ema_edge_open=EMA_EDGE_OPEN,
            ema_edge_close=EMA_EDGE_CLOSE,
            cur_pos=state.rt_position,
        )

        this_size = gen_order_size(
            cur_pos=state.rt_position,
            limit_pos=state.position_limit,
            this_signal=this_signal,
        )
        if this_signal == SignalType.UNLOAD:
            logger.print(f"UNLOAD: {this_signal}, size: {this_size}")
        is_exit = True if this_signal == SignalType.UNLOAD else False
        if this_signal == SignalType.NOTHING:
            return []
        orders = Strategy.smart_place_order(state, order_amt=this_size, is_exit=is_exit)
        return orders

    @staticmethod
    def squid_ink_edgar(state: Status) -> list[Order]:
        mid_price = state.mid
        history_ink = state.hist_mid_prc(150)
        moving_average = np.mean(history_ink)

        # ATR: average absolute difference between consecutive mid prices
        if len(history_ink) >= 2:
            diffs = [
                abs(history_ink[i] - history_ink[i - 1])
                for i in range(1, len(history_ink))
            ]
            diffs = diffs[-30:]
            atr = sum(diffs) / len(diffs)
        else:
            atr = 1e-6  # Prevent division by zero
        raw_feature = mid_price - moving_average
        feature = raw_feature / (atr + 1e-6)

        # Current position
        position = state.rt_position
        orders = []

        # Entry logic
        if abs(position) < 50 and abs(feature) > 0:
            if feature < 0:
                remaining_size = 50 - position
                factor = min(1, -feature / (10))
                scaled_size = int(factor * remaining_size)

                orders.append(
                    Order(
                        state.product,
                        min(state.best_ask - 1, state.best_bid + 1),
                        scaled_size,
                    )
                )  # Buy to open long

            else:
                remaining_size = 50 + position
                factor = min(1, feature / (10))
                scaled_size = int(factor * remaining_size)
                orders.append(
                    Order(
                        state.product,
                        max(state.best_bid + 1, state.best_ask - 1),
                        -scaled_size,
                    )
                )  # Sell to open short

        # Exit logic (mean reversion detected)
        elif position > 0 and feature >= 0:
            orders.append(
                Order(state.product, state.best_bid, -position // 2)
            )  # Sell to close long
        elif position < 0 and feature <= 0:
            orders.append(
                Order(state.product, state.best_ask, -position // 2)
            )  # Buy to close short

        return orders


class Trader:
    state_rainforest_resin = Status("RAINFOREST_RESIN")
    state_kelp = Status("KELP")
    state_squid_ink = Status("SQUID_INK")  # swing but have price trend
    state_croissants = Status("CROISSANTS")
    state_jams = Status("JAMS")
    state_djembes = Status("DJEMBES")
    state_picnic_basket1 = Status("PICNIC_BASKET1")
    state_picnic_basket2 = Status("PICNIC_BASKET2")

    KELP_prices = []
    KELP_vwap = []

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        Status.cls_update(state)
        result = {}

        result["RAINFOREST_RESIN"] = Trade.resin_edgar(self.state_rainforest_resin)
        result["KELP"] = Trade.kelp_edgar(
            self.state_kelp, Trader.KELP_prices, Trader.KELP_vwap
        )
        result["SQUID_INK"] = Trade.squid_ink_edgar(self.state_squid_ink)
        result["PICNIC_BASKET1"] = Trade.picnic_basket1(
            self.state_picnic_basket1,
            self.state_croissants,
            self.state_jams,
            self.state_djembes,
        )
        result["PICNIC_BASKET2"] = []
        result["CROISSANTS"] = []
        result["JAMS"] = []
        result["DJEMBE"] = []

        traderData = "SAMPLE"
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
