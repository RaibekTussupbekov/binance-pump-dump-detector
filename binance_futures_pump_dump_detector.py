import websockets
import json
import asyncio
import logging
import numpy as np
import time
import aiohttp
import argparse

from datetime import datetime
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression
from pathlib import Path

from Binance.rate_limits import RequestWeightLimiterPerSecond
from Binance.rest_api import http_request
from Binance.consts_enums import \
    WEBSOCKET_BASEURL_FUTURES, \
    ENDPOINT_EXCHANGE_INFORMATION_FUTURES, \
    ENDPOINT_CANDLESTICK_DATA_FUTURES, \
    KLINE_CHART_INTERVALS


def parse_command_line_arguments() -> argparse.Namespace:

    """
    Parse command line arguments and put them into parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-z",
        "--hazard",
        type=int,
        default=20,
        help="Number of candlesticks during which 1 pump/dump is expected on average. "
             "Must be at least 5. "
             "Default: 20."
    )
    parser.add_argument(
        "-i",
        "--kline_interval",
        type=str,
        default="5m",
        choices=KLINE_CHART_INTERVALS[:-1],
        help="Candlestick interval: "
             "m -> minutes; h -> hours; d -> days; w -> weeks;. "
             "Default: 5m"
    )
    parser.add_argument(
        "-t",
        "--check_interval",
        type=str,
        default="1m",
        choices=KLINE_CHART_INTERVALS[:-1],
        help="Interval to check for pump/dump: "
             "m -> minutes; h -> hours; d -> days; w -> weeks;. "
             "Default: 1m. "
             "Must be not larger than -i / --kline_interval (Candlestick interval)."
    )
    parser.add_argument(
        "-a",
        "--initial_window",
        type=int,
        default=20,
        help="Initial number of candlesticks to calculate the model parameters. "
             "Must be at least 5. "
             "Default: 20."
    )
    parser.add_argument(
        "-w",
        "--moving_window",
        type=int,
        default=300,
        help="Length of moving window over candlesticks. "
             "Must be at least 2X of -a/--initial_window. "
             "Default: 300."
    )
    parser.add_argument(
        "-c",
        "--changepoint_threshold_probability",
        type=float,
        default=0.5,
        help="Minimal probability to consider the event as pump/dump. "
             "Must be between 0.5 and 0.9 inclusive. "
             "Default: 0.5."
    )
    parser.add_argument(
        "-p",
        "--changepoint_other_path_threshold_probability",
        type=float,
        default=0.01,
        help="Maximal probability of the events other than pump/dump. "
             "Must be between 0.000001 and 0.1 inclusive. "
             "Default: 0.01."
    )

    args = parser.parse_args()

    # arguments validation
    # First, let's ensure that 'check_interval' is not less than 'kline_interval'
    if KLINE_CHART_INTERVALS.index(args.kline_interval) < KLINE_CHART_INTERVALS.index(args.check_interval):
        print("Interval to check for pump/dump (-t / --check_interval must be not less than "
              "candlestick interval (-i / --kline_interval)!")
        raise RuntimeError
    if args.hazard < 5:
        print("Number of candlesticks during which 1 pump/dump is expected on average -z/--hazard. "
              "Must be at least 5!")
        raise RuntimeError
    if args.initial_window < 5:
        print("Initial number of candlesticks to calculate the model parameters -a/--initial_window. "
              "Must be at least 5!")
        raise RuntimeError
    if args.moving_window < (2*args.initial_window):
        print("Length of moving window over candlesticks -w/--moving_window. "
              "Must be at least 2X of -a/--initial_window!")
        raise RuntimeError
    if (args.changepoint_threshold_probability < 0.5) or (args.changepoint_threshold_probability > 0.9):
        print("Minimal probability to consider the event as pump/dump -c/--changepoint_threshold_probability. "
              "Must be between 0.5 and 0.9 inclusive!")
        raise RuntimeError
    if (args.changepoint_other_path_threshold_probability < 1e-6) or \
            (args.changepoint_other_path_threshold_probability > 0.1):
        print("Maximal probability of the events other than pump/dump "
              "-p/--changepoint_other_path_threshold_probability. "
              "Must be between 0.000001 and 0.1 inclusive!")
        raise RuntimeError

    return args


async def exchange_info() -> tuple:

    """

    :return: (
               ALT_COINS: np.ndarray,
               REQUEST_LIMIT_PER_SECOND: int
              )
    """

    async with aiohttp.ClientSession() as client_session:

        status, headers, data = await http_request(
            session=client_session,
            request_weight_limiter=RequestWeightLimiterPerSecond(2400//60),
            endpoint=ENDPOINT_EXCHANGE_INFORMATION_FUTURES
        )

    _alt_coins = []

    for symbol_info in data["symbols"]:

        if (symbol_info["contractType"] == "PERPETUAL") \
                and (symbol_info["status"] == "TRADING") \
                and (symbol_info["quoteAsset"] == "USDT") \
                and (symbol_info["symbol"] != "BTCUSDT"):

            _alt_coins.append(symbol_info["symbol"])

    _request_limit_per_second = 1200 // 60

    for rate_limit in data["rateLimits"]:

        if rate_limit["rateLimitType"] == "REQUEST_WEIGHT":

            # according to the current exchangeInfo, REQUEST_WEIGHT interval can be "MINUTE" only
            _request_limit_per_second = rate_limit["limit"] // rate_limit["intervalNum"] // 60

    return np.array(_alt_coins), _request_limit_per_second


class SimpleLinearRegression:

    def __init__(
            self,
            intercept_init=0.,
            slope_init=1.,
            std_init=1.,
    ):
        """Initialize model.

        p(y) = N(mean, std**2)

        mean = b_0 + b_1 * x

        b_0 ~ N(m_0, s_0**2)
        b_1 ~ N(m_1, s_1**2)
        std ~ Exp(l)

        mean and std are both unknown
        and estimated from (xs, ys) through OLS
        """
        self.intercept_init = intercept_init
        self.slope_init = slope_init
        self.std_init = std_init

        self.intercept_params = np.array([intercept_init])
        self.slope_params = np.array([slope_init])
        self.std_params = np.array([std_init])

    def log_pred_prob(
            self,
            x,
            y
    ):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.intercept_params + self.slope_params * x
        return norm(post_means, self.std_params).logpdf(y)

    def update_params(
            self,
            xs: np.ndarray,
            ys: np.ndarray
    ):
        """Upon observing a new datum (x, y) at time t, update all run length
        hypotheses.
        """

        # run_length = 0
        self.intercept_params = np.array([self.intercept_init])
        self.slope_params = np.array([self.slope_init])
        self.std_params = np.array([self.std_init])

        # run_length = 1
        self.intercept_params = np.append(self.intercept_params, self.intercept_init)

        if xs[-1] != 0:
            self.slope_params = np.append(self.slope_params, (ys[-1] - self.intercept_init) / xs[-1])
        else:
            self.slope_params = np.append(self.slope_params, self.slope_params)

        self.std_params = np.append(self.std_params, self.std_init)

        t = len(xs)

        for run_length in range(2, t + 1):

            lin_reg_model = LinearRegression(n_jobs=-1).fit(xs[t - run_length:t], ys[t - run_length:t])

            self.intercept_params = np.append(self.intercept_params, lin_reg_model.intercept_)
            self.slope_params = np.append(self.slope_params, lin_reg_model.coef_)

            y_pred = lin_reg_model.predict(xs[t - run_length:t])
            std_ = np.std(ys[t - run_length:t] - y_pred)

            # replace zeros with initial std
            if std_ == 0:
                self.std_params = np.append(self.std_params, self.std_init)
            else:
                self.std_params = np.append(self.std_params, std_)


async def get_klines(
        client_session: aiohttp.ClientSession,
        symbol: str,
        limit: int,
        interval: str,
        request_weight_limiter: RequestWeightLimiterPerSecond
) -> dict:

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    status, headers, data = await http_request(
        session=client_session,
        request_weight_limiter=request_weight_limiter,
        endpoint=ENDPOINT_CANDLESTICK_DATA_FUTURES,
        params=params
    )

    return data


async def initialization(
        alt_symbols: np.ndarray,
        kline_interval: str,
        window_init: int,
        hazard: int,
        request_weight_limiter: RequestWeightLimiterPerSecond
) -> tuple:

    # 'xs' is a dictionary with the single 'BTCUSDT' key.
    # Its value is a 2-D numpy array that keeps consecutive percent changes of BTC/USDT pair price.
    #
    # We use dictionary but not directly numpy array to keep 'xs' data
    # because the candlestick stream could fail from time to time and
    # as a consequence the parameter 'xs' of 'candlestick_stream_handler()' is
    # filled again with the initial data.
    #
    # The numpy array shape '(n_samples, n_features)' represents the requirements to
    # 'X' training data of sklearn.linear_model.LinearRegression.fit() method,
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit
    #
    # In the beginning, 'xs' is initialized with data over the last
    # 'window_init + 1' timeframes including the current one.
    xs = {"BTCUSDT": np.zeros((window_init + 1, 1))}

    # 'ys' is a dictionary where each key is an altcoin symbol and
    # each value is a numpy 1-D array that keeps consecutive percent changes of the respected altcoin price.
    ys = {}
    for symbol in alt_symbols:
        ys[symbol] = np.zeros(window_init + 1)

    async with aiohttp.ClientSession() as client_session:

        # first, we get initial data for Simple Linear Regression models
        # including the current timeframe
        limit = window_init+1

        coroutines = [
            get_klines(
                client_session,
                "BTCUSDT",
                limit,
                kline_interval,
                request_weight_limiter
            )
        ]
        for symbol in alt_symbols:
            coroutines.append(
                get_klines(
                    client_session,
                    symbol,
                    limit,
                    kline_interval,
                    request_weight_limiter
                )
            )

        klines = await asyncio.gather(*coroutines)

        open_price_timeframe = {}

        for _i, _kline in enumerate(klines[0]):

            open_price = float(_kline[1])
            close_price = float(_kline[4])

            xs["BTCUSDT"][_i, 0] = (close_price - open_price) * 100 / open_price

        open_price_timeframe["BTCUSDT"] = open_price

        for idx, _klines in enumerate(klines[1:]):

            for _i, _kline in enumerate(_klines):

                open_price = float(_kline[1])
                close_price = float(_kline[4])

                ys[alt_symbols[idx]][_i] = (close_price - open_price) * 100 / open_price

            open_price_timeframe[alt_symbols[idx]] = open_price

    hazard = 1 / hazard
    log_h = np.log(hazard)
    log_1m_h = np.log(1 - hazard)

    model = {}
    log_message = {}

    for symbol in alt_symbols:

        model[symbol] = SimpleLinearRegression()
        log_message[symbol] = np.array([0.])

        for t in range(1, window_init+2):

            # Evaluate predictive probabilities.
            log_pis = model[symbol].log_pred_prob(xs["BTCUSDT"][t-1], ys[symbol][t-1])

            # Calculate growth probabilities.
            log_growth_probs = log_pis + log_message[symbol] + log_1m_h

            # Calculate changepoint probabilities.
            log_cp_prob = logsumexp(log_pis + log_message[symbol] + log_h)

            # Calculate evidence
            new_log_joint = np.append(log_cp_prob, log_growth_probs)

            # Update sufficient statistics.
            model[symbol].update_params(xs["BTCUSDT"][:t], ys[symbol][:t])

            # Pass message.
            log_message[symbol] = new_log_joint

    return xs, ys, model, log_message, open_price_timeframe


def changepoint(
        r_dist,
        changepoint_threshold_prob: float,
        other_path_threshold_prob: float
) -> bool:

    # first, path length falls to 0
    if np.argmax(r_dist) == 1:
        # further, the probability of changepoint must be greater than the set threshold
        if r_dist[1] > changepoint_threshold_prob:
            # and last, all the paths other than changepoint must be improbable
            if np.all(r_dist[4:] < other_path_threshold_prob):
                return True

    return False


async def candlestick_stream_handler(
        websocket,
        kline_interval_minutes: int,
        check_interval_minutes: int,
        window_init: int,
        window: int,
        hazard: int,
        xs: dict,
        ys: dict,
        model: dict,
        log_message: dict,
        changepoint_threshold_prob: float,
        other_path_threshold_prob: float,
        open_price_timeframe: dict
) -> None:

    hazard = 1 / hazard
    log_h = np.log(hazard)
    log_1m_h = np.log(1 - hazard)
    r_dist = {}

    btcusdt_kline_start_time_check_interval_closed = None
    btcusdt_kline_pct_change_check_interval_closed = None
    btcusdt_kline_pct_change = None

    async for message in websocket:

        data = json.loads(message)

        # the first minute of 'kline_interval'
        if (data["data"]["k"]["t"] % (60000 * kline_interval_minutes) == 0) and (data["data"]["k"]["x"]):
            open_price_timeframe[data["data"]["s"]] = float(data["data"]["k"]["o"])

        if data["data"]["s"] in open_price_timeframe:

            if data["data"]["s"] == "BTCUSDT":

                close_price = float(data["data"]["k"]["c"])

                btcusdt_kline_pct_change = (
                                                   close_price - open_price_timeframe["BTCUSDT"]
                                           ) * 100 / open_price_timeframe["BTCUSDT"]

                if check_interval_minutes == 1:
                    check_interval_closed = data["data"]["k"]["x"]
                else:
                    check_interval_closed = (data["data"]["k"]["t"] % (60000 * (check_interval_minutes - 1)) == 0) and \
                                            (data["data"]["k"]["x"])

                if check_interval_closed:
                    btcusdt_kline_start_time_check_interval_closed = data["data"]["k"]["t"]
                    btcusdt_kline_pct_change_check_interval_closed = (
                                                                        close_price - open_price_timeframe["BTCUSDT"]
                                                                     ) * 100 / open_price_timeframe["BTCUSDT"]

            else:

                if check_interval_minutes == 1:
                    check_interval_closed = data["data"]["k"]["x"]
                else:
                    check_interval_closed = (data["data"]["k"]["t"] % (60000 * (check_interval_minutes - 1)) == 0) and \
                                            (data["data"]["k"]["x"])

                if check_interval_closed and (btcusdt_kline_pct_change is not None):

                    close_price = float(data["data"]["k"]["c"])

                    symbol = data["data"]["s"]

                    ys[symbol][-1] = (close_price - open_price_timeframe[symbol]) * 100 / open_price_timeframe[symbol]

                    # we take 'xs[len(ys[symbol]) - 1, 0]', but not 'xs[-1, 0]'
                    # because 'xs' could be already enlarged by 1
                    # in a previous close of the same candlestick of another altcoin

                    # in the case btcusdt check_interval is closed earlier than altcoin check_interval
                    if data["data"]["k"]["t"] == btcusdt_kline_start_time_check_interval_closed:
                        xs["BTCUSDT"][len(ys[symbol]) - 1, 0] = btcusdt_kline_pct_change_check_interval_closed
                    # in the case btcusdt check_interval is closed later than altcoin check_interval
                    else:
                        xs["BTCUSDT"][len(ys[symbol]) - 1, 0] = btcusdt_kline_pct_change

                    # Make model predictions
                    y_pred = None
                    if symbol in r_dist:
                        intercept = np.sum(
                            np.exp(r_dist[symbol][1:]) * model[symbol].intercept_params
                        )
                        slope = np.sum(
                            np.exp(r_dist[symbol][1:]) * model[symbol].slope_params
                        )
                        y_pred = intercept + slope * xs["BTCUSDT"][len(ys[symbol]) - 1, 0]

                    # Evaluate predictive probabilities.
                    log_pis = model[symbol].log_pred_prob(xs["BTCUSDT"][len(ys[symbol]) - 1], ys[symbol][-1])

                    # Calculate growth probabilities.
                    log_growth_probs = log_pis + log_message[symbol] + log_1m_h

                    # Calculate changepoint probabilities.
                    log_cp_prob = logsumexp(log_pis + log_message[symbol] + log_h)

                    # Calculate evidence
                    new_log_joint = np.append(log_cp_prob, log_growth_probs)

                    # Determine run length distribution.
                    r_dist[symbol] = new_log_joint
                    r_dist[symbol] -= logsumexp(new_log_joint)

                    # Changepoint detection
                    if (y_pred is not None) and changepoint(
                            r_dist[symbol], changepoint_threshold_prob, other_path_threshold_prob
                    ):

                        if ys[symbol][-1] > y_pred:
                            print(f"{datetime.now().isoformat()}:  "
                                  f"{data['data']['s']} {ys[symbol][-1]:.3f}% against {y_pred:.3f}% predicted, "
                                  f"BTCUSDT {xs['BTCUSDT'][len(ys[symbol]) - 1, 0]:.3f}% ... Pump?")
                        else:
                            print(f"{datetime.now().isoformat()}:  "
                                  f"{data['data']['s']} {ys[symbol][-1]:.3f}% against {y_pred:.3f}% predicted, "
                                  f"BTCUSDT {xs['BTCUSDT'][len(ys[symbol]) - 1, 0]:.3f}% ... Dump?")

                        logging.info(f"{data['data']['s']}, "
                                     f"{ys[symbol][-1]}, "
                                     f"{y_pred}, "
                                     f"{xs['BTCUSDT'][len(ys[symbol]) - 1, 0]}")

                    # the last minute of kline_interval closed
                    if data["data"]["k"]["t"] % (60000 * kline_interval_minutes) == \
                            (kline_interval_minutes - 1) * 60000:

                        ys[symbol] = np.append(ys[symbol], ys[symbol][-1])

                        # in the case 'xs' has already been enlarged by 1
                        # in a previous close of the same candlestick of another altcoin
                        if len(xs["BTCUSDT"]) < len(ys[symbol]):
                            xs["BTCUSDT"] = np.vstack([xs["BTCUSDT"], xs["BTCUSDT"][-1]])

                        # Pass message.
                        log_message[symbol] = new_log_joint

                        try:
                            r_dist.pop(symbol)
                        except KeyError:
                            pass

                        # clipping the length down to 'initial_window' if it reaches 'moving_window'
                        if len(ys[symbol]) == window:

                            xs["BTCUSDT"] = xs["BTCUSDT"][-window_init:]
                            for _symbol in ys:
                                ys[_symbol] = ys[_symbol][-window_init:]
                            for _symbol in log_message:
                                log_message[_symbol] = log_message[_symbol][-(window_init + 1):]
                            # Update sufficient statistics.
                            for _symbol in model:
                                model[_symbol].update_params(xs["BTCUSDT"], ys[_symbol])
                            for _symbol in r_dist:
                                r_dist[_symbol] = r_dist[_symbol][-(window_init + 2):]

                        else:

                            # Update sufficient statistics.
                            model[symbol].update_params(xs["BTCUSDT"][:len(ys[symbol])], ys[symbol])

                    else:

                        # Update sufficient statistics.
                        model[symbol].update_params(xs["BTCUSDT"][:len(ys[symbol])], ys[symbol])


async def candlestick_stream(
        alt_symbols: np.ndarray,
        kline_interval_minutes: int,
        check_interval_minutes: int,
        window_init: int,
        window: int,
        hazard: int,
        xs: dict,
        ys: dict,
        model: dict,
        log_message: dict,
        changepoint_threshold_prob: float,
        other_path_threshold_prob: float,
        open_price_timeframe: dict
):

    websocket_resource_url = f"{WEBSOCKET_BASEURL_FUTURES}/stream?streams=btcusdt@kline_1m/" \
                             f"{'/'.join([f'{str.lower(symbol)}@kline_1m' for symbol in alt_symbols])}"

    while True:
        try:
            async with websockets.connect(websocket_resource_url) as websocket:
                await candlestick_stream_handler(
                    websocket=websocket,
                    kline_interval_minutes=kline_interval_minutes,
                    check_interval_minutes=check_interval_minutes,
                    window_init=window_init,
                    window=window,
                    hazard=hazard,
                    xs=xs,
                    ys=ys,
                    model=model,
                    log_message=log_message,
                    changepoint_threshold_prob=changepoint_threshold_prob,
                    other_path_threshold_prob=other_path_threshold_prob,
                    open_price_timeframe=open_price_timeframe
                )
        # https://websockets.readthedocs.io/en/stable/faq.html#what-does-connectionclosederror-code-1006-mean
        except websockets.ConnectionClosed:
            continue


def minutes_of_chart_interval(
        chart_interval: str
) -> int:

    time_unit = chart_interval[-1]
    interval_int = int(chart_interval[:-1])
    # minutes
    if time_unit == "m":
        return interval_int
    # hours
    elif time_unit == "h":
        return 60 * interval_int
    # days
    elif time_unit == "d":
        return 24 * 60 * interval_int
    # weeks
    else:
        return 7 * 24 * 60 * interval_int


async def main(
        kline_interval: str,
        check_interval: str,
        window_init: int,
        window: int,
        hazard: int,
        changepoint_threshold_prob: float,
        other_path_threshold_prob: float,
):

    counter_start = time.perf_counter() * 1000

    # alt_coins - list of symbols, other than BTC/USDT
    alt_coins, request_limit_per_second = await exchange_info()

    request_weight_limiter = RequestWeightLimiterPerSecond(request_limit_per_second)

    xs, ys, model, log_message, open_price_timeframe = await initialization(
        alt_symbols=alt_coins,
        kline_interval=kline_interval,
        window_init=window_init,
        hazard=hazard,
        request_weight_limiter=request_weight_limiter
    )
    print(f"initialization took {(time.perf_counter() * 1000 - counter_start):2f}ms...")

    await asyncio.gather(
        candlestick_stream(
            alt_symbols=alt_coins,
            kline_interval_minutes=minutes_of_chart_interval(kline_interval),
            check_interval_minutes=minutes_of_chart_interval(check_interval),
            window_init=window_init,
            window=window,
            hazard=hazard,
            xs=xs,
            ys=ys,
            model=model,
            log_message=log_message,
            changepoint_threshold_prob=changepoint_threshold_prob,
            other_path_threshold_prob=other_path_threshold_prob,
            open_price_timeframe=open_price_timeframe
        )
    )


if __name__ == "__main__":

    command_line_args = parse_command_line_arguments()

    # let's create 'logs' directory if it doesn't exist yet
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        filename=f"logs/binance_futures_pump_dump_detector_"
                 f"i_{command_line_args.kline_interval}_"
                 f"t_{command_line_args.check_interval}_"
                 f"z_{command_line_args.hazard}_"
                 f"a_{command_line_args.initial_window}_"
                 f"w_{command_line_args.moving_window}_"
                 f"c_{command_line_args.changepoint_threshold_probability}_"
                 f"p_{command_line_args.changepoint_other_path_threshold_probability}_"
                 f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}.log",
        level=logging.INFO,
        format="%(asctime)s, %(message)s"
    )

    log_changepoint_threshold_probability = np.log(
        command_line_args.changepoint_threshold_probability
    )
    log_changepoint_other_path_threshold_probability = np.log(
        command_line_args.changepoint_other_path_threshold_probability
    )

    asyncio.run(
        main(
            kline_interval=command_line_args.kline_interval,
            check_interval=command_line_args.check_interval,
            window_init=command_line_args.initial_window,
            window=command_line_args.moving_window,
            hazard=command_line_args.hazard,
            changepoint_threshold_prob=log_changepoint_threshold_probability,
            other_path_threshold_prob=log_changepoint_other_path_threshold_probability,
        )
    )
