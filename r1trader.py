from json import JSONDecoder, JSONEncoder

from datamodel import OrderDepth, Trade, TradingState, Order
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

    POSITION_LIMIT: dict[str, int] = {"AMETHYSTS": 20, "STARFRUIT": 20}

    # length 4, first element is timestamp - 4 iterations, ..., last element is timestamp - 1 iteration
    # B_0 + B_1*predictors[0] + B_2 *predictors[1] + ... + B_4*predictors[3]
    starfruit_predictors: list[float]
    # elements in the form (price, quantity, timestamp)
    # stores data that will be aggregated to a single value, and pushed info starfruit_predictors
    recent_starfruit_trades: list[tuple[int, int, int]]

    def run_AMETHYSTS(self, state: TradingState) -> list[Order]:
        logger = Logger("run_AMETHYSTS", Logger.INFO_LEVEL)
        logger.info("Beginning amethysts trading")

        product = self.AMETHYSTS_NAME
        order_depth: OrderDepth = state.order_depths[product]
        # Initialize the list of Orders to be sent as an empty list
        orders: list[Order] = []

        acceptable_price = 10000

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
        logger.info("Starting starfruit trading")

        product = self.STARFRUIT_NAME
        order_depth: OrderDepth = state.order_depths[product]

        if len(self.starfruit_predictors) < 4:
            return []

        acceptable_price = self.calc_next_trade_price_starfruit()
        logger.info("Acceptable price : " + str(acceptable_price))

        orders: list[Order] = self.calc_buy_and_sell_orders(
            product=product,
            order_depth=order_depth,
            position=state.position.get(product, 0),
            acceptable_price=acceptable_price,
        )

        logger.info(f"Orders: {orders}")

        return orders

    def calc_next_trade_price_starfruit(self) -> int:
        assert len(self.starfruit_predictors) <= 4

        if len(self.starfruit_predictors) == 0:
            # TODO investigate further, 5_000 too low? Maybe set to mid-price
            return 5_000
        elif len(self.starfruit_predictors) < 4:
            return int(np.mean(self.starfruit_predictors))

        # TODO verify coefficients
        # linear regression
        # replace this with coefs B_0, B_1, B_2, B_3, B_4
        betas = np.array(
            [58.2441390858412, 0.18658477, 0.2441745, 0.26626671, 0.2914389]
        )
        # add 1.0 for bias term
        x = np.array([1.0] + self.starfruit_predictors)
        return max(0, int(np.dot(betas, x)))

    def calc_buy_and_sell_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        acceptable_price: int,
    ) -> list[Order]:

        logger = Logger("calc_buy_and_sel_orders", Logger.INFO_LEVEL)

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
                    acceptable_price + 2,
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
                    acceptable_price - 2,
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
                Order(product, acceptable_price - 3, quantity)
            )

        quantity = how_much_to_order(
            quantity_we_can_still_sell, LEFTOVER_QUANTITY_THRESHOLD
        )

        if quantity > 0:
            orders.append(
                # sell order
                Order(product, acceptable_price + 3, -quantity)
            )

        return orders

    def decode_starfruit(
        self,
        decoded_starfruit: Optional[tuple[list[float], list[tuple[int, int, int]]]],
        starfruit_market_trades: list[Trade],
    ):
        logger = Logger("decode_starfruit", Logger.DEBUG_LEVEL)
        logger.info("Decoding STARFRUIT")
        logger.debug(f"decoded_starfruit: {decoded_starfruit}")
        logger.debug(f"starfruit_market_trades: {starfruit_market_trades}")

        if decoded_starfruit == None:
            # first iteration
            self.starfruit_predictors = []
            self.recent_starfruit_trades = []
        else:
            self.starfruit_predictors = decoded_starfruit[0]
            self.recent_starfruit_trades = decoded_starfruit[1]

        logger.debug(f"starfruit_predictors: {self.starfruit_predictors}")
        logger.debug(f"recent_starfruit_trades: {self.recent_starfruit_trades}")

        # check assumptions
        assert len(self.starfruit_predictors) <= 4

        # Get the time stamp corresponding to the trades in self.recent_starfruit_trades at this moment
        last_timestamp_trade_occured = (
            -1
            if len(self.recent_starfruit_trades) == 0
            else self.recent_starfruit_trades[0][2]
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
            self.recent_starfruit_trades = [
                (trade.price, trade.quantity, trade.timestamp)
                for trade in trades_from_timestamp
            ]

            prices_at_timestamp = [trade[0] for trade in self.recent_starfruit_trades]

            if last_timestamp_trade_occured == current_timestamp:
                self.starfruit_predictors.pop()

            self.starfruit_predictors.append(calculate_average(prices_at_timestamp))

            last_timestamp_trade_occured = current_timestamp

            if len(self.starfruit_predictors) > 4:
                self.starfruit_predictors = self.starfruit_predictors[1:]

    def run(self, state: TradingState):

        # ENCODE FORMAT:
        """
        format: A dict whose keys are product names.
        {
            "STARFRUIT": (self.starfruit_cache, self.starfruit_trades)
        }

        """
        traderData: str = state.traderData
        market_trades: dict[str, list[Trade]] = state.market_trades
        logger = Logger("run", Logger.DEBUG_LEVEL)

        ###### STEP 1: DECODE ######
        logger.info("Starting step 1: DECODE")

        decoded = {} if len(traderData) == 0 else JSONDecoder().decode(traderData)

        logger.debug(f"Decoded: {decoded}")

        # dict indexing a key that does not exist throws error
        decoded_starfruit = decoded.get(self.STARFRUIT_NAME, None)
        self.decode_starfruit(
            decoded_starfruit=decoded_starfruit,
            starfruit_market_trades=market_trades.get(self.STARFRUIT_NAME, []),
        )

        ###### STEP 2: PLACE ORDERS #####
        logger.info("Starting step 2: PLACE ORDERS")

        result = {}
        for product in state.order_depths.keys():
            orders: list[Order] = []
            logger.info(f"{product}")

            if product == self.AMETHYSTS_NAME:
                orders = self.run_AMETHYSTS(state)
            elif product == self.STARFRUIT_NAME:
                orders = self.run_STARFRUIT(state)

            result[product] = orders

        ##### STEP 3: ENCODE #####

        traderData = JSONEncoder().encode(
            {
                self.STARFRUIT_NAME: (
                    self.starfruit_predictors,
                    self.recent_starfruit_trades,
                )
            }
        )

        # conversions is in round 2
        conversions = None
        return result, conversions, traderData
