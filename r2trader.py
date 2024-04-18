from collections import OrderedDict
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


def calculate_average(
    a: list[int] | list[float], weights: Optional[list[int]] = None
) -> float:
    if len(a) == 0:
        raise Exception("length of a must be non-zero")

    if weights == None:
        return float(np.mean(a))

    assert len(a) == len(weights)

    return float(np.average(a=a, weights=weights))


def get_best_buy_and_sell_price(order_depth: OrderDepth) -> tuple[int, int]:
    best_buy_price = (
        int(1e7)
        if len(order_depth.buy_orders) == 0
        else max(order_depth.buy_orders.keys())
    )
    best_sell_price = (
        -int(-1e7)
        if len(order_depth.sell_orders) == 0
        else min(order_depth.sell_orders.keys())
    )
    return best_buy_price, best_sell_price


################    END HELPER METHODS     ################


class Trader:
    AMETHYSTS_NAME: str = "AMETHYSTS"
    STARFRUIT_NAME: str = "STARFRUIT"
    ORCHIDS_NAME: str = "ORCHIDS"
    CHOCOLATE_NAME: str = "CHOCOLATE"
    STRAWBERRIES_NAME: str = "STRAWBERRIES"
    ROSES_NAME: str = "ROSES"
    GIFT_BASKET_NAME: str = "GIFT_BASKET"

    POSITION_LIMIT: dict[str, int] = {
        AMETHYSTS_NAME: 20,
        STARFRUIT_NAME: 20,
        ORCHIDS_NAME: 100,
        CHOCOLATE_NAME: 250,
        STRAWBERRIES_NAME: 350,
        ROSES_NAME: 60,
        GIFT_BASKET_NAME: 60,
    }

    ### BEGIN STARFRUIT STATE VAR ###
    P_STARFRUIT = 8  # number of market_price and mid_price predictors (2P in total)
    starfruit_match_price_predictors: list[float]
    # elements of the form (price, quantity, timestamp)
    # stores data that will be aggregated to a single value, and pushed info starfruit_predictors
    recent_starfruit_trades_queue: list[tuple[int, int, int]]
    # Unlike the market trades, mid_prices will only appear once per timestamp, so no need to keep a queue of recent
    starfruit_mid_price_predictors: list[float]
    ### END STARFRUIT STATE VAR ###

    ### BEGIN ORCHIDS STATE VAR ###
    P_ORCHIDS = 1
    orchids_mid_price_predictors: list[float]
    orchids_transport_fees_predictors: list[float]
    orchids_export_tariff_predictors: list[float]
    orchids_import_tariff_predictors: list[float]
    orchids_sunlight_predictors: list[float]
    orchids_humidity_predictors: list[float]
    orchids_iterations_with_long_position: int
    orchids_iterations_with_short_position: int
    ### END ORCHIDS STATE VAR ###

    ### BEGIN CHOCOLATE STATE VAR ###
    P_CHOCOLATE = 4
    chocolate_mid_price_predictors: list[float]
    ### END CHOCOLATE STATE VAR ###

    ### BEGIN STRAWBERRIES STATE VAR ###
    P_STRAWBERRIES = 4
    strawberries_mid_price_predictors: list[float]
    ### END STRAWBERRIES STATE VAR ###

    ### BEGIN ROSES STAT VAR ###
    P_ROSES = 4
    roses_mid_price_predictors: list[float]
    ### END ROSES STATE VAR ###

    ### BEGIN GIFT_BASKET STATE VAR ###
    GIFT_BASKET_BUY_ZSCORE_THRESHOLD = -1.5
    GIFT_BASKET_SELL_ZSCORE_THRESHOLD = 1.5
    GIFT_BASKET_ROLLING_RATIO_WINDOW = 60
    gift_basket_mid_price_predictors: list[float]
    combo_mid_price_predictors: list[float]
    ### END GIFT_BASKET STATE VAR ###

    def run_AMETHYSTS(self, state: TradingState) -> list[Order]:
        logger = Logger("run_AMETHYSTS", Logger.INFO_LEVEL)
        logger.info("Beginning AMETHYSTS trading")

        product = self.AMETHYSTS_NAME
        order_depth: OrderDepth = state.order_depths[product]
        # Initialize the list of Orders to be sent as an empty list
        orders: list[Order] = []

        acceptable_price = 10_000
        logger.info(f"Acceptable price: {str(acceptable_price)}")

        position = state.position.get(product, 0)
        original_position = state.position.get(product, 0)
        position_max = self.POSITION_LIMIT[product]
        position_min = -position_max

        agent_sell_orders = OrderedDict(sorted(list(order_depth.sell_orders.items())))
        agent_buy_orders = OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True)
        )

        best_buy_price, best_sell_price = get_best_buy_and_sell_price(
            order_depth=order_depth
        )

        # Considering the offers available to us (and knowing they didn't match),
        # lets try to set offers that are more enticing than the ones offered to us
        generous_buy_price = best_buy_price + 1
        generous_sell_price = best_sell_price - 1

        ### OUR BUY ORDERS ###
        for ask_price, ask_vol in agent_sell_orders.items():
            # we are looking to increase our position (i.e. we are buying more)
            if position < position_max and (
                ask_price < acceptable_price
                or (ask_price == acceptable_price and position < 0)
            ):
                # Don't go over the limit
                order_vol = min(abs(ask_vol), position_max - position)
                assert order_vol > 0
                orders.append(Order(product, ask_price, order_vol))
                position += order_vol

        if position < position_max and original_position < 0:
            # We want to make sure that we get out of a negative position and into a positive now
            # Be a bit more generous here
            order_vol = position_max - position
            assert order_vol > 0
            orders.append(
                Order(
                    product,
                    min(generous_buy_price + 1, acceptable_price - 1),
                    order_vol,
                )
            )
            position += order_vol

        if position < position_max and original_position > 15:
            # Don't be too generous and offer out best possible deal, we were already near the long position
            order_vol = position_max - position
            assert order_vol > 0
            orders.append(
                Order(product, min(best_buy_price, acceptable_price - 1), order_vol)
            )
            position += order_vol

        if position < position_max:
            order_vol = position_max - position
            assert order_vol > 0
            orders.append(
                Order(product, min(generous_buy_price, acceptable_price - 1), order_vol)
            )
            position += order_vol

        ### OUR SELL ORDERS ###
        position = original_position  # reset position limit

        for bid_price, bid_vol in agent_buy_orders.items():
            # we are looking to decrease our position (i.e. we are selling our inventory)
            if position > position_min and (
                bid_price > acceptable_price
                or (bid_price == acceptable_price and position > 0)
            ):
                # Don't go over the limit
                order_vol = min(abs(bid_vol), position - position_min)
                assert order_vol > 0
                orders.append(Order(product, bid_price, -order_vol))
                position -= order_vol

        if position > position_min and original_position > 0:
            # We want to make sure that we get out of a positive position and into a negative now
            # Be a bit more generous here
            order_vol = position - position_min
            assert order_vol > 0
            orders.append(
                Order(
                    product,
                    max(generous_sell_price - 1, acceptable_price + 1),
                    -order_vol,
                )
            )
            position -= order_vol

        if position > position_min and original_position < -15:
            # Don't be too generous and offer out best possible deal, we were already near the short position
            order_vol = position - position_min
            assert order_vol > 0
            orders.append(
                Order(product, max(best_sell_price, acceptable_price + 1), -order_vol)
            )
            position -= order_vol

        if position > position_min:
            order_vol = position - position_min
            assert order_vol > 0
            orders.append(
                Order(
                    product, max(generous_sell_price, acceptable_price + 1), -order_vol
                )
            )
            position -= order_vol

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
        conversionObservation: Optional[ConversionObservation] = (
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

    def run_CHOCOLATE(self, state: TradingState) -> list[Order]:

        if len(self.chocolate_mid_price_predictors) == 0:
            return []

        return self.get_CHOCOLATE_orders(
            order_depth=state.order_depths.get(self.CHOCOLATE_NAME, OrderDepth()),
            position=state.position.get(self.CHOCOLATE_NAME, 0),
            acceptable_price=self.estimate_fair_price_chocolate(),
        )

    def run_STRAWBERRIES(self, state: TradingState) -> list[Order]:
        if len(self.strawberries_mid_price_predictors) == 0:
            return []

        return self.get_STRAWBERRIES_orders(
            order_depth=state.order_depths.get(self.STRAWBERRIES_NAME, OrderDepth()),
            position=state.position.get(self.STRAWBERRIES_NAME, 0),
            acceptable_price=self.estimate_fair_price_strawberries(),
        )

    def run_ROSES(self, state: TradingState) -> list[Order]:
        if len(self.roses_mid_price_predictors) == 0:
            return []

        return self.get_ROSES_orders(
            order_depth=state.order_depths.get(self.ROSES_NAME, OrderDepth()),
            position=state.position.get(self.ROSES_NAME, 0),
            acceptable_price=self.estimate_fair_price_roses(),
        )

    def run_GIFT_BASKET(self, state: TradingState) -> list[Order]:
        if len(self.gift_basket_mid_price_predictors) == 0:
            return []

        return self.get_GIFT_BASKET_orders(
            order_depth=state.order_depths.get(self.GIFT_BASKET_NAME, OrderDepth()),
            position=state.position.get(self.GIFT_BASKET_NAME, 0),
            z_score=self.get_z_score_gift_baskets(),
        )

    def estimate_fair_price_starfruit(self) -> int:
        assert len(self.starfruit_match_price_predictors) == self.P_STARFRUIT
        assert len(self.starfruit_mid_price_predictors) == self.P_STARFRUIT

        # TODO verify coefficients
        # Linear regression

        beta = np.array(
            [
                8.964154125777895,
                0.00401325,
                -0.01097109,
                0.00503411,
                0.00219081,
                0.01628759,
                0.00836369,
                -0.00575337,
                -0.01325094,
                0.02501386,
                0.03937632,
                0.0703923,
                0.09963095,
                0.09229896,
                0.16020264,
                0.21157717,
                0.29381952,
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
            + self.orchids_mid_price_predictors
            + self.orchids_transport_fees_predictors
            + self.orchids_export_tariff_predictors
            + self.orchids_import_tariff_predictors
            + self.orchids_sunlight_predictors
            + self.orchids_humidity_predictors
        )

        return max(0, int(np.dot(beta, x)))

    def estimate_fair_price_chocolate(self) -> int:
        assert len(self.chocolate_mid_price_predictors) >= 1

        if len(self.chocolate_mid_price_predictors) < self.P_CHOCOLATE:
            return int(calculate_average(self.chocolate_mid_price_predictors))

        assert len(self.chocolate_mid_price_predictors) == self.P_CHOCOLATE

        beta = np.array(
            [0.14240089710074244, -0.01108489, 0.00786715, 0.04070572, 0.96249293]
        )

        x = np.array(
            [1.0] + self.chocolate_mid_price_predictors  # add 1.0 for intercept term
        )

        return max(0, int(np.dot(beta, x)))

    def estimate_fair_price_strawberries(self) -> int:
        assert len(self.strawberries_mid_price_predictors) >= 1

        if len(self.strawberries_mid_price_predictors) < self.P_STRAWBERRIES:
            return int(calculate_average(self.strawberries_mid_price_predictors))

        assert len(self.strawberries_mid_price_predictors) == self.P_STRAWBERRIES

        beta = np.array(
            [0.37947875425743405, -0.00548191, 0.05122488, 0.1366758, 0.81748684]
        )

        x = np.array(
            [1.0] + self.strawberries_mid_price_predictors  # add 1.0 for intercept term
        )

        return max(0, int(np.dot(beta, x)))

    def estimate_fair_price_roses(self) -> int:
        assert len(self.roses_mid_price_predictors) >= 1

        if len(self.roses_mid_price_predictors) < self.P_ROSES:
            return int(calculate_average(self.roses_mid_price_predictors))

        assert len(self.roses_mid_price_predictors) == self.P_ROSES

        beta = np.array(
            [4.238897536935838, -0.00885487, 0.00141759, 0.01359939, 0.99354437]
        )

        x = np.array(
            [1.0] + self.roses_mid_price_predictors  # add 1.0 for intercept term
        )

        return max(0, int(np.dot(beta, x)))

    def get_z_score_gift_baskets(self) -> float:
        logger = Logger("get_z_score_gift_baskets", Logger.DEBUG_LEVEL)
        logger.info("Computing zscore for gift baskets rolling ratio")
        assert len(self.gift_basket_mid_price_predictors) >= 1

        if (
            len(self.gift_basket_mid_price_predictors)
            < self.GIFT_BASKET_ROLLING_RATIO_WINDOW
        ):
            logger.info(
                f"There were less than {self.GIFT_BASKET_ROLLING_RATIO_WINDOW} gift basket mid prices computed. Returning z_score: 0"
            )
            return 0.0

        assert (
            len(self.gift_basket_mid_price_predictors)
            == self.GIFT_BASKET_ROLLING_RATIO_WINDOW
        )
        assert (
            len(self.combo_mid_price_predictors)
            == self.GIFT_BASKET_ROLLING_RATIO_WINDOW
        )

        gift_basket = np.array(self.gift_basket_mid_price_predictors)
        combo = np.array(self.combo_mid_price_predictors)

        ratio = gift_basket / combo

        logger.debug(
            f"Past {self.GIFT_BASKET_ROLLING_RATIO_WINDOW} ratios: {ratio.tolist()}"
        )

        # TODO investigate if changing 5 to some other number gives better results
        z_score: float = (ratio[-5:].mean() - ratio.mean()) / (ratio.std() + 1e-8)

        logger.info(f"z_score: {z_score}")

        return z_score

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
        logger.info("Generating ORCHIDS orders and conversions")

        product = self.ORCHIDS_NAME
        orders: list[Order] = []
        conversions: int = 0
        position_max = self.POSITION_LIMIT[product]
        position_min = -1 * self.POSITION_LIMIT[product]
        UNIT_ORCHID_STORAGE_COST = 0.1

        if conversionObservation == None:
            logger.warn("conversionObservation is none")
            return [], 0

        logger.info(f"Conversion bid price {conversionObservation.bidPrice}")
        logger.info(f"Conversion ask price {conversionObservation.askPrice}")

        south_bid_price_after_fees = (
            conversionObservation.bidPrice
            - conversionObservation.exportTariff
            - conversionObservation.transportFees
            - UNIT_ORCHID_STORAGE_COST
        )

        south_ask_price_after_fees = (
            conversionObservation.askPrice
            + conversionObservation.importTariff
            + conversionObservation.transportFees
        )

        logger.info(f"South bid price after fees: {south_bid_price_after_fees}")
        logger.info(f"South ask price after fees: {south_ask_price_after_fees}")

        # Conversions
        logger.info("Generating ORCHIDS conversions (if possible)")

        if conversionObservation != None:

            # BEGIN Logging
            if abs(conversionObservation.importTariff) > abs(
                conversionObservation.exportTariff
            ):
                logger.warn(
                    f"Import tariff = {conversionObservation.importTariff} is GREATER than export tariff = {conversionObservation.exportTariff} (in absolute value)"
                )
            # END Logging

            # BEGIN conversion logic
            LONG_POSITION_ITERATIONS_BEFORE_FORCE_CONVERSION = 15
            SHORT_POSITION_ITERATIONS_BEFORE_FORCE_CONVERSION = 20

            FORCE_CONVERSION_FROM_LONG: bool = (
                self.orchids_iterations_with_long_position
                >= LONG_POSITION_ITERATIONS_BEFORE_FORCE_CONVERSION
            )
            FORCE_CONVERSION_FROM_SHORT: bool = (
                self.orchids_iterations_with_short_position
                >= SHORT_POSITION_ITERATIONS_BEFORE_FORCE_CONVERSION
            )

            if position != 0:
                logger.info(f"Position is non-zero. Analyzing conversion observation.")
                if position > 0:
                    agent_sell_orders = OrderedDict(
                        sorted(order_depth.sell_orders.items())
                    )
                    convert_amount = 0

                    for ask_price, ask_vol in agent_sell_orders.items():
                        # we are looking to increase our position after decreasing our position from this conversion
                        if (
                            ask_price < south_bid_price_after_fees
                            and convert_amount < abs(position)
                        ):
                            # Don't go over the limit
                            order_vol = min(
                                abs(ask_vol), abs(position) - convert_amount
                            )
                            # orders.append(Order(product, ask_price, order_vol))
                            convert_amount += order_vol

                    if convert_amount > 0:
                        conversions -= abs(convert_amount)
                        logger.info(
                            f"Joshua's Case -- Conversion: Selling {abs(convert_amount)} orchid(s) @ agent's bid price of {conversionObservation.bidPrice}"
                        )

                    if abs(conversions) < abs(position) and (
                        south_bid_price_after_fees >= acceptable_price
                        or FORCE_CONVERSION_FROM_LONG
                    ):
                        # Decreasing our position
                        convert_amount_remaining = abs(position) - abs(conversions)
                        logger.info(
                            f"Conversion: Selling {abs(convert_amount_remaining)} orchid(s) @ agent's bid price of {conversionObservation.bidPrice}; FORCE_CONVERSION: {FORCE_CONVERSION_FROM_LONG}"
                        )
                        conversions -= abs(convert_amount_remaining)

                    position -= abs(conversions)

                elif position < 0:
                    agent_buy_orders = OrderedDict(
                        sorted(order_depth.buy_orders.items(), reverse=True)
                    )
                    convert_amount = 0

                    for bid_price, bid_vol in agent_buy_orders.items():
                        # we are looking to decrease our position after increasing our position from this conversion
                        if (
                            bid_price > south_ask_price_after_fees
                            and convert_amount < abs(position)
                        ):
                            # Don't go over the limit
                            order_vol = min(
                                abs(bid_vol), abs(position) - convert_amount
                            )
                            # orders.append(Order(product, bid_price, -order_vol))
                            convert_amount += order_vol

                    if convert_amount > 0:
                        conversions += convert_amount
                        logger.info(
                            f"Joshua's Case -- Conversion: Buying {abs(convert_amount)} orchid(s) @ agent's ask price of {conversionObservation.askPrice}"
                        )

                    if abs(conversions) < abs(position) and (
                        south_ask_price_after_fees < acceptable_price
                        or FORCE_CONVERSION_FROM_SHORT
                    ):
                        convert_amount_remaining = abs(position) - abs(conversions)
                        # Increasing our position
                        logger.info(
                            f"Conversion: Buying {abs(convert_amount_remaining)} orchid(s) @ agent's ask price of {conversionObservation.askPrice}; FORCE_CONVERSION: {FORCE_CONVERSION_FROM_SHORT}"
                        )
                        conversions += abs(convert_amount_remaining)

                    position += abs(conversions)

            # We are assuming here that either conversion is zero or
            # if non-zero it will take us back to position 0
            if conversions != 0:
                self.orchids_iterations_with_long_position = 0
                self.orchids_iterations_with_short_position = 0
            elif conversions == 0 and position != 0:
                if position > 0:
                    self.orchids_iterations_with_long_position += 1
                    self.orchids_iterations_with_short_position = 0
                else:
                    self.orchids_iterations_with_long_position = 0
                    self.orchids_iterations_with_short_position += 1
            else:
                # conversion == position == 0
                self.orchids_iterations_with_long_position = 0
                self.orchids_iterations_with_short_position = 0

            # END conversion logic

        logger.info("Generating ORCHIDS orders")

        # keep all values denoting volume/quantity non-negative, until we place actually place orders

        best_sell_price = (
            -int(1e8)
            if len(order_depth.sell_orders) == 0
            else min(order_depth.sell_orders.keys())
        )
        best_buy_price = (
            int(1e8)
            if len(order_depth.buy_orders) == 0
            else min(order_depth.buy_orders.keys())
        )
        logger.debug(
            f"best_sell_price: {best_sell_price}, best_buy_price: {best_buy_price}"
        )

        agent_sell_orders_we_considering = [
            abs(vol)
            for price, vol in order_depth.sell_orders.items()
            if price < int(south_bid_price_after_fees)
        ]
        agent_buy_orders_we_considering = [
            abs(vol)
            for price, vol in order_depth.buy_orders.items()
            if price > int(south_ask_price_after_fees)
        ]

        agent_sell_orders_we_considering_quantity = sum(
            agent_sell_orders_we_considering
        )
        agent_buy_orders_we_considering_quantity = sum(agent_buy_orders_we_considering)

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
            # sell order
            orders.append(
                Order(
                    product,
                    int(south_ask_price_after_fees + 1),
                    -our_sell_quantity,
                )
            )

        if our_buy_quantity != 0:
            # buy order
            orders.append(
                Order(
                    product,
                    int(south_bid_price_after_fees - 1),
                    our_buy_quantity,
                )
            )

        # how much more we can still sell in current order
        quantity_we_can_still_sell = abs(position_low - position_min)
        # how much more we can still buy in current order
        quantity_we_can_still_buy = abs(position_high - position_max)

        LEFTOVER_QUANTITY_THRESHOLD = 0

        # Handle the leftovers after adjusting for being too near position limit

        quantity = how_much_to_order(
            quantity_we_can_still_buy, LEFTOVER_QUANTITY_THRESHOLD
        )

        if quantity > 0:
            orders.append(
                # buy order
                Order(
                    product,
                    max(best_buy_price, int(south_bid_price_after_fees) - 2),
                    quantity,
                )
            )

        quantity = how_much_to_order(
            quantity_we_can_still_sell, LEFTOVER_QUANTITY_THRESHOLD
        )

        if quantity > 0:
            orders.append(
                # sell order
                Order(
                    product,
                    min(best_sell_price, int(south_ask_price_after_fees) + 2),
                    -quantity,
                )
            )

        logger.info(f"Orders: {orders}")
        logger.info(f"Conversion: {conversions}")

        return orders, conversions

    def get_CHOCOLATE_orders(
        self, order_depth: OrderDepth, position: int, acceptable_price: int
    ) -> list[Order]:
        logger = Logger("get_CHOCOLATE_orders", Logger.DEBUG_LEVEL)
        logger.info("Generating CHOCOLATE orders")

        orders: list[Order] = []

        logger.info(f"Orders: {orders}")
        return orders

    def get_STRAWBERRIES_orders(
        self, order_depth: OrderDepth, position: int, acceptable_price: int
    ) -> list[Order]:
        logger = Logger("get_STRAWBERRIES_orders", Logger.DEBUG_LEVEL)
        logger.info("Generating STRAWBERRIES orders")

        orders: list[Order] = []

        logger.info(f"Orders: {orders}")
        return orders

    def get_ROSES_orders(
        self, order_depth: OrderDepth, position: int, acceptable_price: int
    ) -> list[Order]:
        logger = Logger("get_ROSES_orders", Logger.DEBUG_LEVEL)
        logger.info("Generating ROSES orders")

        orders: list[Order] = []
        logger.info(f"Orders: {orders}")
        return orders

    def get_GIFT_BASKET_orders(
        self, order_depth: OrderDepth, position: int, z_score: float
    ) -> list[Order]:
        logger = Logger("get_GIFT_BASKET_orders", Logger.DEBUG_LEVEL)
        logger.info("Generating GIFT_BASKET orders")

        position_min = -1 * self.POSITION_LIMIT[self.GIFT_BASKET_NAME]
        position_max = self.POSITION_LIMIT[self.GIFT_BASKET_NAME]

        orders: list[Order] = []

        # greater z_score means sell
        if z_score >= self.GIFT_BASKET_SELL_ZSCORE_THRESHOLD:
            agent_buy_orders = OrderedDict(
                sorted(order_depth.buy_orders.items(), reverse=True)
            )

            # Iterate through agent buy orders
            for buy_price, buy_vol in agent_buy_orders.items():
                our_sell_vol = min(
                    abs(buy_vol),
                    abs(position_min - position),
                )
                if our_sell_vol == 0:
                    break
                orders.append(
                    Order(self.GIFT_BASKET_NAME, buy_price, -1 * our_sell_vol)
                )
                position -= our_sell_vol

            # Left overs
            our_sell_vol_remaining = abs(position_min - position)
            if our_sell_vol_remaining > 0:
                generous_sell_price = int(self.gift_basket_mid_price_predictors[-1]) + 1
                actual_sell_vol = (
                    (our_sell_vol_remaining + 1) // 2
                    if z_score < 2
                    else our_sell_vol_remaining
                )
                orders.append(
                    Order(
                        self.GIFT_BASKET_NAME,
                        generous_sell_price,
                        -1 * actual_sell_vol,
                    )
                )
                position -= actual_sell_vol

        # lower z_score means buy
        if z_score <= self.GIFT_BASKET_BUY_ZSCORE_THRESHOLD:
            agent_sell_orders = OrderedDict(
                sorted(list(order_depth.sell_orders.items()))
            )

            for sell_price, sell_vol in agent_sell_orders.items():
                our_buy_vol = min(
                    abs(sell_vol),
                    abs(position_max - position),
                )
                if our_buy_vol == 0:
                    break
                orders.append(
                    Order(self.GIFT_BASKET_NAME, sell_price, abs(our_buy_vol))
                )
                position += our_buy_vol

            # Left overs
            our_buy_vol_remaining = abs(position_max - position)
            if our_buy_vol_remaining > 0:
                generous_buy_price = int(self.gift_basket_mid_price_predictors[-1]) - 1
                actual_buy_vol = (
                    (our_buy_vol_remaining + 1) // 2
                    if z_score > -2
                    else our_buy_vol_remaining
                )
                orders.append(
                    Order(
                        self.GIFT_BASKET_NAME,
                        generous_buy_price,
                        actual_buy_vol,
                    )
                )
                position += actual_buy_vol

        logger.info(f"Orders: {orders}")
        return orders

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

        if decoded_starfruit == None:
            # first iteration
            self.starfruit_match_price_predictors = []
            self.recent_starfruit_trades_queue = []
            self.starfruit_mid_price_predictors = []
        else:
            self.starfruit_match_price_predictors = decoded_starfruit[0]
            self.recent_starfruit_trades_queue = decoded_starfruit[1]
            self.starfruit_mid_price_predictors = decoded_starfruit[2]

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
            logger.debug(f"market trades current timestamp: {current_timestamp}")
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

        logger.debug(
            f"starfruit_market_price_predictors: {self.starfruit_match_price_predictors}"
        )
        logger.debug(
            f"recent_starfruit_trades_queue: {self.recent_starfruit_trades_queue}"
        )
        logger.debug(
            f"starfruit_mid_price_predictors: {self.starfruit_mid_price_predictors}"
        )

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
                int,
                int,
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
            self.orchids_iterations_with_long_position = 0
            self.orchids_iterations_with_short_position = 0
        else:
            self.orchids_mid_price_predictors = decoded_orchids[0]
            self.orchids_transport_fees_predictors = decoded_orchids[1]
            self.orchids_export_tariff_predictors = decoded_orchids[2]
            self.orchids_import_tariff_predictors = decoded_orchids[3]
            self.orchids_sunlight_predictors = decoded_orchids[4]
            self.orchids_humidity_predictors = decoded_orchids[5]
            self.orchids_iterations_with_long_position = decoded_orchids[6]
            self.orchids_iterations_with_short_position = decoded_orchids[7]

        # check assumptions
        expected_common_len = len(self.orchids_mid_price_predictors)
        assert expected_common_len <= self.P_ORCHIDS
        assert len(self.orchids_mid_price_predictors) == expected_common_len
        assert len(self.orchids_transport_fees_predictors) == expected_common_len
        assert len(self.orchids_export_tariff_predictors) == expected_common_len
        assert len(self.orchids_import_tariff_predictors) == expected_common_len
        assert len(self.orchids_sunlight_predictors) == expected_common_len
        assert len(self.orchids_humidity_predictors) == expected_common_len

        if conversionObservation == None:
            logger.warn("Did not recieve a conservation observation for orchids")
            return

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
        logger.debug(
            f"iterations_with_long_position: {self.orchids_iterations_with_long_position}"
        )
        logger.debug(
            f"iterations_with_short_position: {self.orchids_iterations_with_short_position}"
        )

    def decode_chocolate(
        self,
        mid_price_predictors: Optional[list[float]],
        chocolate_order_depth: OrderDepth,
    ):
        logger = Logger("decode_chocolate", Logger.DEBUG_LEVEL)
        logger.info("Decoding CHOCOLATE")

        if mid_price_predictors == None:
            self.chocolate_mid_price_predictors = []
        else:
            self.chocolate_mid_price_predictors = mid_price_predictors

        if (
            len(chocolate_order_depth.buy_orders) != 0
            and len(chocolate_order_depth.sell_orders) != 0
        ):
            mid_price = calculate_average(
                list(chocolate_order_depth.buy_orders.keys())
                + list(chocolate_order_depth.sell_orders.keys())
            )

            self.chocolate_mid_price_predictors.append(mid_price)

        if len(self.chocolate_mid_price_predictors) > self.P_CHOCOLATE:
            self.chocolate_mid_price_predictors.pop(0)

        logger.debug(
            f"chocolate_mid_price_predictors: {self.chocolate_mid_price_predictors}"
        )

    def decode_strawberries(
        self,
        mid_price_predictors: Optional[list[float]],
        strawberries_order_depth: OrderDepth,
    ):
        logger = Logger("decode_strawberries", Logger.DEBUG_LEVEL)
        logger.info("Decoding STRAWBERRIES")

        if mid_price_predictors == None:
            self.strawberries_mid_price_predictors = []
        else:
            self.strawberries_mid_price_predictors = mid_price_predictors

        if (
            len(strawberries_order_depth.buy_orders) != 0
            and len(strawberries_order_depth.sell_orders) != 0
        ):
            mid_price = calculate_average(
                list(strawberries_order_depth.buy_orders.keys())
                + list(strawberries_order_depth.sell_orders.keys())
            )

            self.strawberries_mid_price_predictors.append(mid_price)

        if len(self.strawberries_mid_price_predictors) > self.P_STRAWBERRIES:
            self.strawberries_mid_price_predictors.pop(0)

        logger.debug(
            f"strawberries_mid_price_predictors: {self.strawberries_mid_price_predictors}"
        )

    def decode_roses(
        self, mid_price_predictors: Optional[list[float]], roses_order_depth: OrderDepth
    ):
        logger = Logger("decode_roses", Logger.DEBUG_LEVEL)
        logger.info("Decoding ROSES")

        if mid_price_predictors == None:
            self.roses_mid_price_predictors = []
        else:
            self.roses_mid_price_predictors = mid_price_predictors

        if (
            len(roses_order_depth.buy_orders) != 0
            and len(roses_order_depth.sell_orders) != 0
        ):
            mid_price = calculate_average(
                list(roses_order_depth.buy_orders.keys())
                + list(roses_order_depth.sell_orders.keys())
            )

            self.roses_mid_price_predictors.append(mid_price)

        if len(self.roses_mid_price_predictors) > self.P_ROSES:
            self.roses_mid_price_predictors.pop(0)

        logger.debug(f"roses_mid_price_predictors: {self.roses_mid_price_predictors}")

    def decode_gift_basket(
        self,
        decoded_gift_basket: Optional[tuple[list[float], list[float]]],
        gift_basket_order_depth: OrderDepth,
        chocolate_order_depth: OrderDepth,
        strawberries_order_depth: OrderDepth,
        roses_order_depth: OrderDepth,
    ):
        logger = Logger("decode_gift_basket", Logger.DEBUG_LEVEL)
        logger.info("Decoding GIFT_BASKET")

        if decoded_gift_basket == None:
            self.gift_basket_mid_price_predictors = []
            self.combo_mid_price_predictors = []
        else:
            self.gift_basket_mid_price_predictors = decoded_gift_basket[0]
            self.combo_mid_price_predictors = decoded_gift_basket[1]

        best_buy_price, best_sell_price = get_best_buy_and_sell_price(
            gift_basket_order_depth
        )
        gift_basket_mid_price = (best_buy_price + best_sell_price) / 2

        # Add mid price
        self.gift_basket_mid_price_predictors.append(gift_basket_mid_price)

        combo_mid_price = 0

        for order_depth, weight in zip(
            [chocolate_order_depth, strawberries_order_depth, roses_order_depth],
            [4, 6, 1],
        ):
            best_buy_price, best_sell_price = get_best_buy_and_sell_price(
                order_depth=order_depth
            )
            combo_mid_price += weight * (best_buy_price + best_sell_price) / 2

        # add combo mid price
        self.combo_mid_price_predictors.append(combo_mid_price)

        # Pop mid price
        if (
            len(self.gift_basket_mid_price_predictors)
            > self.GIFT_BASKET_ROLLING_RATIO_WINDOW
        ):
            self.gift_basket_mid_price_predictors.pop(0)

        # Pop combo mid price
        if len(self.combo_mid_price_predictors) > self.GIFT_BASKET_ROLLING_RATIO_WINDOW:
            self.combo_mid_price_predictors.pop(0)

        logger.debug(
            f"gift_basket_mid_price_predictors: {self.gift_basket_mid_price_predictors}"
        )

        logger.debug(f"combo_mid_price_predictors: {self.combo_mid_price_predictors}")

    def run(self, state: TradingState):
        traderData: str = state.traderData
        market_trades: dict[str, list[Trade]] = state.market_trades
        conversionObservations = state.observations.conversionObservations
        timestamp = state.timestamp
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

        decoded_chocolate = decoded.get(self.CHOCOLATE_NAME, None)
        self.decode_chocolate(
            decoded_chocolate,
            state.order_depths.get(self.CHOCOLATE_NAME, OrderDepth()),
        )

        decoded_strawberries = decoded.get(self.STRAWBERRIES_NAME, None)
        self.decode_strawberries(
            decoded_strawberries,
            state.order_depths.get(self.STRAWBERRIES_NAME, OrderDepth()),
        )

        decoded_roses = decoded.get(self.ROSES_NAME, None)
        self.decode_roses(
            decoded_roses, state.order_depths.get(self.ROSES_NAME, OrderDepth())
        )

        decoded_gift_basket = decoded.get(self.GIFT_BASKET_NAME, None)

        self.decode_gift_basket(
            decoded_gift_basket=decoded_gift_basket,
            gift_basket_order_depth=state.order_depths.get(
                self.GIFT_BASKET_NAME, OrderDepth()
            ),
            chocolate_order_depth=state.order_depths.get(
                self.CHOCOLATE_NAME, OrderDepth()
            ),
            strawberries_order_depth=state.order_depths.get(
                self.STRAWBERRIES_NAME, OrderDepth()
            ),
            roses_order_depth=state.order_depths.get(self.ROSES_NAME, OrderDepth()),
        )

        ###### STEP 2: PLACE ORDERS #####
        logger.info("STEP 2: PLACE ORDERS")

        result = {}
        conversions = None

        for product in state.order_depths.keys():
            orders: list[Order] = []
            logger.info(f"Position = {state.position.get(product, 0)} for {product}")
            print(
                f"72b8f0c1-bdb8-42d2-81c6-ca32bbb0a6b0,{timestamp},{product},{state.position.get(product, 0)}"
            )
            logger.info(f"{product} own trades:  {state.own_trades.get(product, [])}")
            logger.info(
                f"{product} market trades:  {state.market_trades.get(product, [])}"
            )

            if product == self.AMETHYSTS_NAME:
                pass
                # orders = self.run_AMETHYSTS(state)
            elif product == self.STARFRUIT_NAME:
                pass
                # orders = self.run_STARFRUIT(state)
            elif product == self.ORCHIDS_NAME:
                pass
                # orders, conversions = self.run_ORCHIDS(state)
            elif product == self.CHOCOLATE_NAME:
                pass
                # orders = self.run_CHOCOLATE(state)
            elif product == self.STRAWBERRIES_NAME:
                pass
                # orders = self.run_STRAWBERRIES(state)
            elif product == self.ROSES_NAME:
                pass
                # orders = self.run_ROSES(state)
            elif product == self.GIFT_BASKET_NAME:
                orders = self.run_GIFT_BASKET(state)

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
                    self.orchids_iterations_with_long_position,
                    self.orchids_iterations_with_short_position,
                ),
                self.CHOCOLATE_NAME: self.chocolate_mid_price_predictors,
                self.STRAWBERRIES_NAME: self.strawberries_mid_price_predictors,
                self.ROSES_NAME: self.roses_mid_price_predictors,
                self.GIFT_BASKET_NAME: (
                    self.gift_basket_mid_price_predictors,
                    self.combo_mid_price_predictors,
                ),
            }
        )

        return result, conversions, traderData
