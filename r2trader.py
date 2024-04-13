from json import JSONDecoder, JSONEncoder

from datamodel import ConversionObservation, OrderDepth, Trade, TradingState, Order
from typing import Optional
import numpy as np


class Logger:
    INFO_LEVEL = 0
    WARN_LEVEL = 1
    DEBUG_LEVEL = 2

    def __init__(self, name: str, level: int):
        self.name_ = name
        self.level_ = level

    def info(self, message: str):
        if self.level_ >= Logger.INFO_LEVEL:
            print(f"[INFO][{self.name_}] {message}", flush=True)

    def warn(self, message: str):
        if self.level_ >= Logger.WARN_LEVEL:
            print(f"[WARN][{self.name_}] {message}", flush=True)

    def debug(self, message: str):
        if self.level_ >= Logger.DEBUG_LEVEL:
            print(f"[DEBUG][{self.name_}] {message}", flush=True)


################    BEGIN HELPER METHODS     ################


def how_much_to_order(
    pre_can_still_trade_quantity: int, post_can_still_trade_quantity: int
) -> int:
    return max(pre_can_still_trade_quantity - post_can_still_trade_quantity, 0)


def calculate_average(a: list[int], weights: Optional[list[int]] = None) -> float:
    if len(a) == 0:
        raise Exception("length of a must be non-zero")

    if weights == None:
        return float(np.mean(a))

    assert len(a) == len(weights)

    return float(np.average(a=a, weights=weights))


################    END HELPER METHODS     ################


class Trader:
    AMETHYSTS_NAME: str = "AMETHYSTS"
    STARFRUIT_NAME: str = "STARFRUIT"
    ORCHIDS_NAME: str = "ORCHIDS"

    POSITION_LIMIT: dict[str, int] = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100}

    ### BEGIN STARFRUIT STATE VAR ###
    P_STARFRUIT = 4  # number of market_price and mid_price predictors (2P in total)
    starfruit_match_price_predictors: list[float]
    # elements of the form (price, quantity, timestamp)
    # stores data that will be aggregated to a single value, and pushed info starfruit_predictors
    recent_starfruit_trades_queue: list[tuple[int, int, int]]
    # Unlike the market trades, mid_prices will only appear once per timestamp, so no need to keep a queue of recent
    starfruit_mid_price_predictors: list[float]
    ### END STARFRUIT STATE VAR ###

    ### BEGIN ORCHIDS STATE VAR ###
    P_ORCHIDS = 4
    orchids_mid_price_predictors: list[float]
    orchids_transport_fees_predictors: list[float]
    orchids_export_tariff_predictors: list[float]
    orchids_import_tariff_predictors: list[float]
    orchids_sunlight_predictors: list[float]
    orchids_humidity_predictors: list[float]
    ### END ORCHIDS STATE VAR ###

    def run_AMETHYSTS(self, state: TradingState) -> list[Order]:
        logger = Logger("run_AMETHYSTS", Logger.INFO_LEVEL)
        logger.info("Beginning AMETHYSTS trading")

        product = self.AMETHYSTS_NAME
        order_depth: OrderDepth = state.order_depths[product]
        # Initialize the list of Orders to be sent as an empty list
        orders: list[Order] = []

        acceptable_price = 10_000

        logger.info(f"Acceptable price: {str(acceptable_price)}")

        best_sell = 0
        sell_quantity = 0
        best_buy = 0
        buy_quantity = 0

        # Order depth list come already sorted.
        # We can simply pick first item to check first item to get best bid or offer
        if len(order_depth.sell_orders) != 0:
            best_sell = min(order_depth.sell_orders.keys())

            filtered_sell_orders = [
                sell_order
                for sell_order in order_depth.sell_orders.items()
                if sell_order[0] < acceptable_price
            ]

            filtered_quantities = [
                abs(quantity) for _, quantity in filtered_sell_orders
            ]

            sell_quantity = sum(filtered_quantities)

        if len(order_depth.buy_orders) != 0:
            best_buy = max(order_depth.buy_orders.keys())

            filtered_buy_orders = [
                buy_order
                for buy_order in order_depth.buy_orders.items()
                if buy_order[0] > acceptable_price
            ]

            filtered_quantities = [abs(quantity) for _, quantity in filtered_buy_orders]
            buy_quantity = sum(filtered_quantities)

        position_current = state.position.get(product, 0)

        logger.debug(f"Current position = {position_current} for product = {product}")

        position_max = self.POSITION_LIMIT[product]
        position_min = -position_max

        buy_quantity = min(buy_quantity, abs(position_current - position_min))
        sell_quantity = min(sell_quantity, abs(position_current - position_max))
        intersect_quantity = min(buy_quantity, sell_quantity)

        buy_quantity -= intersect_quantity
        sell_quantity -= intersect_quantity

        if buy_quantity != 0:
            if abs(position_min - position_current) < 5:
                # near short limit
                buy_quantity = 0
        elif sell_quantity != 0:
            if abs(position_max - position_current) < 5:
                # near long limit
                sell_quantity = 0

        assert position_min <= position_current - intersect_quantity - buy_quantity
        assert position_max >= position_current + intersect_quantity + sell_quantity

        orders.append(Order(product, best_buy, -intersect_quantity - buy_quantity))
        orders.append(Order(product, best_sell, intersect_quantity + sell_quantity))

        position_min_diff = abs(
            position_current - intersect_quantity - buy_quantity - position_min
        )
        position_max_diff = abs(
            position_current + intersect_quantity + sell_quantity - position_max
        )

        # orders above what the bots usually offer
        orders.append(Order(product, int(1e4) + 4, -position_min_diff))
        orders.append(Order(product, int(1e4) - 4, position_max_diff))

        assert (
            position_min
            <= position_current - intersect_quantity - buy_quantity - position_min_diff
        )
        assert (
            position_max
            >= position_current + intersect_quantity + sell_quantity + position_max_diff
        )

        logger.debug(f"Orders: {orders}")

        return orders

    def run_STARFRUIT(self, state: TradingState) -> list[Order]:
        logger = Logger("run_STARFRUIT", Logger.DEBUG_LEVEL)
        logger.info("Beginning STARFRUIT trading")

        order_depth: OrderDepth = state.order_depths[self.STARFRUIT_NAME]

        logger.debug(f"OrderDepth Buy: {order_depth.buy_orders}")
        logger.debug(f"OrderDepth Sell: {order_depth.sell_orders}")

        # TODO investigate if we can estimate bid price without regression
        if (
            len(self.starfruit_match_price_predictors) < self.P_STARFRUIT
            or len(self.starfruit_mid_price_predictors) < self.P_STARFRUIT
        ):
            return []

        acceptable_price = self.estimate_fair_price_starfruit()
        logger.info("Acceptable price: " + str(acceptable_price))

        orders: list[Order] = self.get_orders_STARFRUIT(
            order_depth=order_depth,
            position=state.position.get(self.STARFRUIT_NAME, 0),
            acceptable_price=acceptable_price,
        )

        logger.info(f"Orders: {orders}")

        return orders

    def run_ORCHIDS(self, state: TradingState) -> tuple[list[Order], int]:
        logger = Logger("run_ORCHIDS", Logger.DEBUG_LEVEL)
        logger.info("Beginning ORCHIDS trading")

        order_depth: OrderDepth = state.order_depths[self.ORCHIDS_NAME]
        conversionObservation: ConversionObservation = (
            state.observations.conversionObservations.get(self.ORCHIDS_NAME, None)
        )

        logger.debug(f"OrderDepth Buy: {order_depth.buy_orders}")
        logger.debug(f"OrderDepth Sell: {order_depth.sell_orders}")

        if len(self.orchids_mid_price_predictors) < self.P_ORCHIDS:
            return [], 0

        acceptable_price = self.estimate_fair_price_orchids()
        logger.info("Estimated next orchids price: " + str(acceptable_price))

        return self.get_ORCHIDS_orders_and_conversions(
            order_depth=order_depth,
            position=state.position.get(self.ORCHIDS_NAME, 0),
            acceptable_price=acceptable_price,
            conversionObservation=conversionObservation,
        )

    def estimate_fair_price_starfruit(self) -> int:
        assert len(self.starfruit_match_price_predictors) == self.P_STARFRUIT
        assert len(self.starfruit_mid_price_predictors) == self.P_STARFRUIT

        # TODO verify coefficients
        # Linear regression

        beta = np.array(
            [
                26.44753348300128,
                0.03073808,
                0.02727196,
                0.02748499,
                -0.02022061,
                0.15585123,
                0.19657682,
                0.2511217,
                0.32589726,
            ]
        )

        x = np.array(
            [1.0]  # add 1.0 for intercept term
            + self.starfruit_match_price_predictors
            + self.starfruit_mid_price_predictors
        )

        return max(0, int(np.dot(beta, x)))

    def estimate_fair_price_orchids(self) -> int:

        assert len(self.orchids_mid_price_predictors) == self.P_ORCHIDS
        assert len(self.orchids_transport_fees_predictors) == self.P_ORCHIDS
        assert len(self.orchids_export_tariff_predictors) == self.P_ORCHIDS
        assert len(self.orchids_import_tariff_predictors) == self.P_ORCHIDS
        assert len(self.orchids_sunlight_predictors) == self.P_ORCHIDS
        assert len(self.orchids_humidity_predictors) == self.P_ORCHIDS

        beta = np.array(
            [
                0.06987954,
                9.99711443e-01,
                1.09677694e-02,
                6.46992420e-03,
                -2.80465300e-03,
                1.75439458e-05,
                1.37126605e-03,
            ]
        )

        x = np.array(
            [1.0]  # add 1.0 for intercept term
            + self.orchids_mid_price_predictors[-1:]
            + self.orchids_transport_fees_predictors[-1:]
            + self.orchids_export_tariff_predictors[-1:]
            + self.orchids_import_tariff_predictors[-1:]
            + self.orchids_sunlight_predictors[-1:]
            + self.orchids_humidity_predictors[-1:]
        )

        return max(0, int(np.dot(beta, x)))

    def get_orders_STARFRUIT(
        self,
        order_depth: OrderDepth,
        position: int,
        acceptable_price: int,
    ) -> list[Order]:
        logger = Logger("get_orders_STARFRUIT", Logger.INFO_LEVEL)
        logger.info("Generating STARFRUIT orders")

        product = self.STARFRUIT_NAME

        # keep all values denoting volume/quantity non-negative, until we place actually place orders
        agent_sell_orders_we_considering = [
            abs(vol)
            for price, vol in order_depth.sell_orders.items()
            if price < acceptable_price
        ]
        agent_buy_orders_we_considering = [
            abs(vol)
            for price, vol in order_depth.buy_orders.items()
            if price > acceptable_price
        ]

        agent_sell_orders_we_considering_quantity = sum(
            agent_sell_orders_we_considering
        )
        agent_buy_orders_we_considering_quantity = sum(agent_buy_orders_we_considering)

        # one of these values must be zero
        assert (
            agent_sell_orders_we_considering_quantity == 0
            or agent_buy_orders_we_considering_quantity == 0
        )

        logger.debug(f"position = {position} for product = {product}")

        position_max = self.POSITION_LIMIT[product]
        position_min = -1 * self.POSITION_LIMIT[product]

        # don't buy more than your position limits
        our_sell_quantity = min(
            agent_buy_orders_we_considering_quantity, abs(position - position_min)
        )
        our_buy_quantity = min(
            agent_sell_orders_we_considering_quantity, abs(position - position_max)
        )

        position_low = position - our_sell_quantity
        position_high = position + our_buy_quantity
        assert position_min <= position_low <= position_high <= position_max

        orders: list[Order] = []
        if our_sell_quantity != 0:
            orders.append(
                # sell order
                Order(
                    product,
                    acceptable_price + 1,
                    -our_sell_quantity,
                )
            )
        if our_buy_quantity != 0:
            orders.append(
                # buy order
                Order(
                    product,
                    acceptable_price - 1,
                    our_buy_quantity,
                )
            )

        # how much more we can still sell in current order
        quantity_we_can_still_sell = abs(position_low - position_min)
        # how much more we can still buy in current order
        quantity_we_can_still_buy = abs(position_high - position_max)

        NEAR_LIMIT_THRESHOLD = 2
        LEFTOVER_QUANTITY_THRESHOLD = 4

        # Prevent being stuck at position limits

        if (
            quantity_we_can_still_sell <= NEAR_LIMIT_THRESHOLD
            and quantity_we_can_still_buy != 0
        ):
            # near short limit
            # TODO this logic only works for products with position limits 20
            quantity = min(quantity_we_can_still_buy, 5)

            orders.append(
                # buy order
                Order(
                    product,
                    acceptable_price - 1,
                    quantity,
                )
            )
            quantity_we_can_still_buy -= quantity

        if (
            quantity_we_can_still_buy <= NEAR_LIMIT_THRESHOLD
            and quantity_we_can_still_sell != 0
        ):
            # near long limit
            # TODO this logic only works for products with position limits 20
            quantity = min(quantity_we_can_still_sell, 5)

            orders.append(
                # sell order
                Order(
                    product,
                    acceptable_price + 1,
                    -quantity,
                )
            )
            quantity_we_can_still_sell -= quantity

        # Handle the leftovers after adjusting for being too near position limit

        quantity = how_much_to_order(
            quantity_we_can_still_buy, LEFTOVER_QUANTITY_THRESHOLD
        )

        if quantity > 0:
            orders.append(
                # buy order
                Order(product, acceptable_price - 2, quantity)
            )

        quantity = how_much_to_order(
            quantity_we_can_still_sell, LEFTOVER_QUANTITY_THRESHOLD
        )

        if quantity > 0:
            orders.append(
                # sell order
                Order(product, acceptable_price + 2, -quantity)
            )

        return orders

    def get_ORCHIDS_orders_and_conversions(
        self,
        order_depth: OrderDepth,
        position: int,
        acceptable_price: int,
        conversionObservation: Optional[ConversionObservation],
    ) -> tuple[list[Order], int]:
        logger = Logger("get_ORCHIDS_orders_and_conversions", Logger.DEBUG_LEVEL)

        product = self.ORCHIDS_NAME
        orders: list[Order] = []
        conversions: int = 0

        UNIT_ORCHID_STORAGE_COST = 0.1

        if abs(conversionObservation.importTariff) > abs(
            conversionObservation.exportTariff
        ):
            logger.warn(
                f"Import tariff = {conversionObservation.importTariff} is GREATER than export tariff = {conversionObservation.exportTariff} (in absolute value)"
            )

        adjusted_acceptable_agent_bid_price = (
            conversionObservation.bidPrice
            - conversionObservation.exportTariff
            - conversionObservation.transportFees
            - UNIT_ORCHID_STORAGE_COST
        )

        adjusted_acceptable_agent_ask_price = (
            conversionObservation.askPrice
            + conversionObservation.importTariff
            + conversionObservation.transportFees
        )

        logger.info(
            f"Adjusted fair agent bid price: {adjusted_acceptable_agent_bid_price}"
        )
        logger.info(
            f"Adjusted fair agent bid price: {adjusted_acceptable_agent_ask_price}"
        )

        # Conversions
        logger.info("Generating ORCHIDS conversions if possible")
        if conversionObservation != None and position != 0:
            logger.info(
                f"Position is non-zero. Analyzing conversion observation with acceptable price {acceptable_price}."
            )
            if position > 0 and adjusted_acceptable_agent_bid_price >= acceptable_price:
                # We will sell to the agent
                logger.info(
                    f"Conversion: Selling {abs(position)} orchid(s) @ agent's bid price of {conversionObservation.askPrice}"
                )
                conversions = -1 * abs(position)
                position = 0

            elif (
                position < 0 and adjusted_acceptable_agent_ask_price < acceptable_price
            ):
                # We will buy from the agent
                logger.info(
                    f"Conversion: Buying {abs(position)} orchid(s) @ agent's ask price of {conversionObservation.askPrice}"
                )
                conversions = abs(position)
                position = 0

        logger.info("Generating ORCHIDS orders")

        # keep all values denoting volume/quantity non-negative, until we place actually place orders

        agent_sell_orders_we_considering = [
            abs(vol)
            for price, vol in order_depth.sell_orders.items()
            if price < adjusted_acceptable_agent_ask_price
        ]
        agent_buy_orders_we_considering = [
            abs(vol)
            for price, vol in order_depth.buy_orders.items()
            if price >= adjusted_acceptable_agent_bid_price
        ]

        agent_sell_orders_we_considering_quantity = sum(
            agent_sell_orders_we_considering
        )
        agent_buy_orders_we_considering_quantity = sum(agent_buy_orders_we_considering)

        # one of these values must be zero
        assert (
            agent_sell_orders_we_considering_quantity == 0
            or agent_buy_orders_we_considering_quantity == 0
        )

        logger.debug(f"position = {position} for product = {product}")

        position_max = self.POSITION_LIMIT[product]
        position_min = -1 * self.POSITION_LIMIT[product]

        # don't buy more than your position limits
        our_sell_quantity = min(
            agent_buy_orders_we_considering_quantity, abs(position - position_min)
        )
        our_buy_quantity = min(
            agent_sell_orders_we_considering_quantity, abs(position - position_max)
        )

        position_low = position - our_sell_quantity
        position_high = position + our_buy_quantity
        assert position_min <= position_low <= position_high <= position_max

        orders: list[Order] = []
        if our_sell_quantity != 0:
            orders.append(
                # sell order
                Order(
                    product,
                    acceptable_price + 1,
                    -our_sell_quantity,
                )
            )
        if our_buy_quantity != 0:
            orders.append(
                # buy order
                Order(
                    product,
                    acceptable_price - 1,
                    our_buy_quantity,
                )
            )

        # how much more we can still sell in current order
        quantity_we_can_still_sell = abs(position_low - position_min)
        # how much more we can still buy in current order
        quantity_we_can_still_buy = abs(position_high - position_max)

        # LEFTOVER_QUANTITY_THRESHOLD = 0

        # # Handle the leftovers after adjusting for being too near position limit

        # quantity = how_much_to_order(
        #     quantity_we_can_still_buy, LEFTOVER_QUANTITY_THRESHOLD
        # )

        # if quantity > 0:
        #     orders.append(
        #         # buy order
        #         Order(product, acceptable_price - 2, quantity)
        #     )

        # quantity = how_much_to_order(
        #     quantity_we_can_still_sell, LEFTOVER_QUANTITY_THRESHOLD
        # )

        # if quantity > 0:
        #     orders.append(
        #         # sell order
        #         Order(product, acceptable_price + 2, -quantity)
        #     )

        logger.info(f"Orders: {orders}")
        logger.info(f"Conversion: {conversions}")

        return orders, conversions

    def decode_starfruit(
        self,
        decoded_starfruit: Optional[
            tuple[list[float], list[tuple[int, int, int]], list[float]]
        ],
        starfruit_market_trades: list[Trade],
        starfruit_order_depths: OrderDepth,
    ):
        logger = Logger("decode_starfruit", Logger.DEBUG_LEVEL)
        logger.info("Decoding STARFRUIT")
        logger.debug(f"decoded_starfruit: {decoded_starfruit}")
        logger.debug(f"starfruit_market_trades: {starfruit_market_trades}")

        if decoded_starfruit == None:
            # first iteration
            self.starfruit_match_price_predictors = []
            self.recent_starfruit_trades_queue = []
            self.starfruit_mid_price_predictors = []
        else:
            self.starfruit_match_price_predictors = decoded_starfruit[0]
            self.recent_starfruit_trades_queue = decoded_starfruit[1]
            self.starfruit_mid_price_predictors = decoded_starfruit[2]

        logger.debug(
            f"starfruit_market_price_predictors: {self.starfruit_match_price_predictors}"
        )
        logger.debug(
            f"recent_starfruit_trades_queue: {self.recent_starfruit_trades_queue}"
        )
        logger.debug(
            f"starfruit_mid_price_predictors: {self.starfruit_mid_price_predictors}"
        )

        # check assumptions
        assert len(self.starfruit_match_price_predictors) <= self.P_STARFRUIT
        assert len(self.starfruit_mid_price_predictors) <= self.P_STARFRUIT

        # Get the time stamp corresponding to the trades in self.recent_starfruit_trades at this moment
        last_timestamp_trade_occured = (
            -1
            if len(self.recent_starfruit_trades_queue) == 0
            else self.recent_starfruit_trades_queue[0][2]
        )

        """
        Want to handle this case: same market trades span consecutive timestamps
        {
            "market_trades: {'STARFRUIT': [(STARFRUIT,  << , 5003.0, 1, 4300)], ...}
            "timestamp": 4300
        }
        {
            "market_trades: {'STARFRUIT': [(STARFRUIT,  << , 5003.0, 1, 4300), (STARFRUIT,  << , 4998.0, 2, 4300)], ...}
            "timestamp": 4400
        }
        """

        # SUBSTEP 1: Update market price predictors

        for current_timestamp in sorted(
            set([trade.timestamp for trade in starfruit_market_trades])
        ):
            logger.debug(f"current_timestamp: {current_timestamp}")
            # Will always be non-empty
            trades_from_timestamp = [
                trade
                for trade in starfruit_market_trades
                if trade.timestamp == current_timestamp
            ]

            # Re-assign recent starfruit trades
            self.recent_starfruit_trades_queue = [
                (trade.price, trade.quantity, trade.timestamp)
                for trade in trades_from_timestamp
            ]

            prices_at_timestamp = [
                trade[0] for trade in self.recent_starfruit_trades_queue
            ]

            if last_timestamp_trade_occured == current_timestamp:
                self.starfruit_match_price_predictors.pop()

            self.starfruit_match_price_predictors.append(
                calculate_average(prices_at_timestamp)
            )

            last_timestamp_trade_occured = current_timestamp

            if len(self.starfruit_match_price_predictors) > self.P_STARFRUIT:
                self.starfruit_match_price_predictors.pop(0)

        # SUBSTEP 2: Update mid price predictors

        if (
            len(starfruit_order_depths.buy_orders) != 0
            and len(starfruit_order_depths.sell_orders) != 0
        ):
            best_buy_price = max(starfruit_order_depths.buy_orders.keys())
            best_sell_price = min(starfruit_order_depths.sell_orders.keys())

            self.starfruit_mid_price_predictors.append(
                (best_buy_price + best_sell_price) / 2
            )

        if len(self.starfruit_mid_price_predictors) > self.P_STARFRUIT:
            self.starfruit_mid_price_predictors.pop(0)

    def decode_orchids(
        self,
        decoded_orchids: Optional[
            tuple[
                list[float],
                list[float],
                list[float],
                list[float],
                list[float],
                list[float],
            ]
        ],
        conversionObservation: Optional[ConversionObservation],
    ):
        logger = Logger("decode_orchids", Logger.DEBUG_LEVEL)

        if decoded_orchids == None:
            # first iteration
            self.orchids_mid_price_predictors = []
            self.orchids_transport_fees_predictors = []
            self.orchids_export_tariff_predictors = []
            self.orchids_import_tariff_predictors = []
            self.orchids_sunlight_predictors = []
            self.orchids_humidity_predictors = []
        else:
            self.orchids_mid_price_predictors = decoded_orchids[0]
            self.orchids_transport_fees_predictors = decoded_orchids[1]
            self.orchids_export_tariff_predictors = decoded_orchids[2]
            self.orchids_import_tariff_predictors = decoded_orchids[3]
            self.orchids_sunlight_predictors = decoded_orchids[4]
            self.orchids_humidity_predictors = decoded_orchids[5]

        if conversionObservation == None:
            logger.warn("Did not recieve a conservation observation for orchids")
            return

        logger.debug(f"mid_price_predictors: {self.orchids_mid_price_predictors}")
        logger.debug(
            f"transport_fees_predictors: {self.orchids_transport_fees_predictors}"
        )
        logger.debug(
            f"export_tariff_predictors: {self.orchids_export_tariff_predictors}"
        )
        logger.debug(
            f"import_tariff_predictors: {self.orchids_import_tariff_predictors}"
        )
        logger.debug(f"sunlight_predictors: {self.orchids_sunlight_predictors}")
        logger.debug(f"humidity_predictors: {self.orchids_humidity_predictors}")

        # check assumptions
        expected_common_len = len(self.orchids_mid_price_predictors)
        assert expected_common_len <= self.P_ORCHIDS
        assert len(self.orchids_mid_price_predictors) == expected_common_len
        assert len(self.orchids_transport_fees_predictors) == expected_common_len
        assert len(self.orchids_export_tariff_predictors) == expected_common_len
        assert len(self.orchids_import_tariff_predictors) == expected_common_len
        assert len(self.orchids_sunlight_predictors) == expected_common_len
        assert len(self.orchids_humidity_predictors) == expected_common_len

        askPrice = conversionObservation.askPrice
        bidPrice = conversionObservation.bidPrice
        transportFees = conversionObservation.transportFees
        exportTariff = conversionObservation.exportTariff
        importTariff = conversionObservation.importTariff
        sunlight = conversionObservation.sunlight
        humidity = conversionObservation.humidity

        if expected_common_len >= self.P_ORCHIDS:
            self.orchids_mid_price_predictors.pop(0)
            self.orchids_transport_fees_predictors.pop(0)
            self.orchids_export_tariff_predictors.pop(0)
            self.orchids_import_tariff_predictors.pop(0)
            self.orchids_sunlight_predictors.pop(0)
            self.orchids_humidity_predictors.pop(0)

        self.orchids_mid_price_predictors.append((askPrice + bidPrice) / 2)
        self.orchids_transport_fees_predictors.append(transportFees)
        self.orchids_export_tariff_predictors.append(exportTariff)
        self.orchids_import_tariff_predictors.append(importTariff)
        self.orchids_sunlight_predictors.append(sunlight)
        self.orchids_humidity_predictors.append(humidity)

    def run(self, state: TradingState):

        # ENCODE FORMAT:
        """
        format: A dict whose keys are product names.
        {
            "STARFRUIT": (match_price_predictors, recent_match_prices_queue, mid_price_predictors)
            "ORCHIDS": (mid_price_predictors, transport_fees_predictors, export_tariff_predictors, import_tariff_predictors, sunlight_predictors, humidity_predictors)
        }

        """
        traderData: str = state.traderData
        market_trades: dict[str, list[Trade]] = state.market_trades
        conversionObservations = state.observations.conversionObservations
        logger = Logger("run", Logger.DEBUG_LEVEL)

        ###### STEP 1: DECODE ######
        logger.info("STEP 1: DECODE")

        decoded = {} if len(traderData) == 0 else JSONDecoder().decode(traderData)

        logger.debug(f"Decoded: {decoded}")

        # dict indexing a key that does not exist throws error
        decoded_starfruit = decoded.get(self.STARFRUIT_NAME, None)
        self.decode_starfruit(
            decoded_starfruit=decoded_starfruit,
            starfruit_market_trades=market_trades.get(self.STARFRUIT_NAME, []),
            starfruit_order_depths=state.order_depths.get(
                self.STARFRUIT_NAME, OrderDepth()
            ),
        )

        decoded_orchids = decoded.get(self.ORCHIDS_NAME, None)
        self.decode_orchids(
            decoded_orchids=decoded_orchids,
            conversionObservation=conversionObservations.get(self.ORCHIDS_NAME, None),
        )

        ###### STEP 2: PLACE ORDERS #####
        logger.info("STEP 2: PLACE ORDERS")

        result = {}
        conversions = None

        for product in state.order_depths.keys():
            orders: list[Order] = []
            logger.info(f"{product}")

            if product == self.AMETHYSTS_NAME:
                orders = self.run_AMETHYSTS(state)
            elif product == self.STARFRUIT_NAME:
                orders = self.run_STARFRUIT(state)
            elif product == self.ORCHIDS_NAME:
                orders, conversions = self.run_ORCHIDS(state)

            result[product] = orders

        ##### STEP 3: ENCODE #####
        logger.info("STEP 3: ENCODE")

        traderData = JSONEncoder().encode(
            {
                self.STARFRUIT_NAME: (
                    self.starfruit_match_price_predictors,
                    self.recent_starfruit_trades_queue,
                    self.starfruit_mid_price_predictors,
                ),
                self.ORCHIDS_NAME: (
                    self.orchids_mid_price_predictors,
                    self.orchids_transport_fees_predictors,
                    self.orchids_export_tariff_predictors,
                    self.orchids_import_tariff_predictors,
                    self.orchids_sunlight_predictors,
                    self.orchids_humidity_predictors,
                ),
            }
        )

        # conversions is in round 2
        return result, conversions, traderData