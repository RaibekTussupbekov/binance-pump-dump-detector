from enum import Enum
from datetime import datetime, timezone
import os

# Based on the format of 'Date' header in HTTP response
# e.g. 'Fri, 02 Oct 2020 10:10:08 GMT'
DATETIME_FORMAT = '%a, %d %b %Y %H:%M:%S %Z'

# https://binance-docs.github.io/apidocs/futures/en/#general-api-information
BASE_ENDPOINT_FUTURES = "https://fapi.binance.com"

# https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
WEBSOCKET_BASEURL_FUTURES = "wss://fstream.binance.com"

# https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams#general-wss-information
WEBSOCKET_BASEURL_SPOT = "wss://stream.binance.com:9443"

# https://binance-docs.github.io/apidocs/spot/en/#general-info
BASE_ENDPOINT_SPOT = "https://api.binance.com"

# 2019-09-08T17:57:00 is the first moment of trading on Binance Futures
FIRST_TRADING_DATETIME = datetime(2019, 9, 8, 17, 57, tzinfo=timezone.utc)
# in milliseconds
FIRST_TRADING_TIME_MILLIS = 1567965420000

# API Key with restrictions:
# * CAN READ
try:
    API_KEY_READ_ONLY = os.environ['API_KEY_READ_ONLY']
    SECRET_KEY_READ_ONLY = os.environ['SECRET_KEY_READ_ONLY']
except KeyError:
    API_KEY_READ_ONLY = ''
    SECRET_KEY_READ_ONLY = ''

# API Key with restrictions:
# * CAN READ
# * Enable Spot & Margin Trading
# * Enable Margin
# * Enable Futures
try:
    API_KEY_TRADING = os.environ['API_KEY_TRADING']
    SECRET_KEY_TRADING = os.environ['SECRET_KEY_TRADING']
except KeyError:
    API_KEY_TRADING = ''
    SECRET_KEY_TRADING = ''

# API Key with restrictions:
# * CAN READ
# * Enable Withdrawals
try:
    API_KEY_WITHDRAWALS = os.environ['API_KEY_WITHDRAWALS']
    SECRET_KEY_WITHDRAWALS = os.environ['SECRET_KEY_WITHDRAWALS']
except KeyError:
    API_KEY_WITHDRAWALS = ''
    SECRET_KEY_WITHDRAWALS = ''

# possible Kline/Candlestick chart intervals of Kline/Candlestick Streams
# https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-streams
KLINE_CHART_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]


class RequestMethod(Enum):
    GET = "get"
    POST = "post"
    PUT = "put"
    DELETE = "delete"


class EndpointSecurityType(Enum):
    """
    https://binance-docs.github.io/apidocs/futures/en/#endpoint-security-type
    https://binance-docs.github.io/apidocs/spot/en/#endpoint-security-type
    """

    NONE = "Endpoint can be accessed freely."
    TRADE = "Endpoint requires sending a valid API - Key and signature."
    MARGIN = "Endpoint requires sending a valid API-Key and signature."
    USER_DATA = "Endpoint requires sending a valid API - Key and signature."
    USER_STREAM = "Endpoint requires sending a valid API - Key."
    MARKET_DATA = "Endpoint requires sending a valid API - Key."

    # Extra security type aimed only for withdrawals
    WITHDRAWALS = "Endpoint requires sending a valid API - Key and signature."


class Endpoint:

    def __init__(
            self,
            base_endpoint: str,
            request_method: RequestMethod,
            endpoint: str,
            endpoint_security_type: EndpointSecurityType,
            weight_method
    ):
        assert callable(weight_method)
        assert base_endpoint in (BASE_ENDPOINT_FUTURES, BASE_ENDPOINT_SPOT)
        self.base_endpoint = base_endpoint
        self.request_method = request_method
        self.endpoint = endpoint
        self.endpoint_security_type = endpoint_security_type
        self.get_weight = weight_method


# ******************* Binance Futures Endpoints ********************** #

# https://binance-docs.github.io/apidocs/futures/en/#test-connectivity
ENDPOINT_TEST_CONNECTIVITY_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/ping",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#check-server-time
ENDPOINT_CHECK_SERVER_TIME_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/time",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#exchange-information
ENDPOINT_EXCHANGE_INFORMATION_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/exchangeInfo",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)


def weight_endpoint_order_book_futures(params={}) -> int:
    if "limit" not in params:
        return 10
    assert params["limit"] in [5, 10, 20, 50, 100, 500, 1000]
    if params["limit"] == 100:
        return 5
    elif params["limit"] == 500:
        return 10
    elif params["limit"] == 1000:
        return 20
    else:
        return 2


# https://binance-docs.github.io/apidocs/futures/en/#order-book
ENDPOINT_ORDER_BOOK_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/depth",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=weight_endpoint_order_book_futures
)

# https://binance-docs.github.io/apidocs/futures/en/#recent-trades-list
ENDPOINT_RECENT_TRADES_LIST_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/trades",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#old-trades-lookup-market_data
ENDPOINT_OLD_TRADES_LOOKUP_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/historicalTrades",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 5
)

# https://binance-docs.github.io/apidocs/futures/en/#compressed-aggregate-trades-list
ENDPOINT_COMPRESSED_TRADES_LIST_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/aggTrades",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)


def weight_endpoint_candlestick_data_futures(params={}) -> int:
    if "limit" not in params:
        return 5
    assert (params["limit"] >= 1) and (params["limit"] <= 1500)
    if (params["limit"] >= 1) and (params["limit"] < 100):
        return 1
    elif (params["limit"] >= 100) and (params["limit"] < 500):
        return 2
    elif (params["limit"] >= 500) and (params["limit"] <= 1000):
        return 5
    else:
        return 10


# https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
ENDPOINT_CANDLESTICK_DATA_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/klines",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=weight_endpoint_candlestick_data_futures
)

# https://binance-docs.github.io/apidocs/futures/en/#mark-price
ENDPOINT_MARK_PRICE_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/premiumIndex",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
ENDPOINT_GET_FUNDING_RATE_HISTORY_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/fundingRate",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)


def weight_endpoint_24hr_ticker_change_statistics_futures(params={}) -> int:
    if "symbol" in params:
        return 40
    else:
        return 1


# https://binance-docs.github.io/apidocs/futures/en/#24hr-ticker-price-change-statistics
ENDPOINT_24HR_TICKER_CHANGE_STATISTICS_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/ticker/24hr",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=weight_endpoint_24hr_ticker_change_statistics_futures
)


def weight_endpoint_symbol_price_ticker_futures(params={}) -> int:
    if "symbol" in params:
        return 2
    else:
        return 1


# https://binance-docs.github.io/apidocs/futures/en/#symbol-price-ticker
ENDPOINT_SYMBOL_PRICE_TICKER_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/ticker/price",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=weight_endpoint_symbol_price_ticker_futures
)


def weight_endpoint_symbol_order_book_ticker_futures(params={}) -> int:
    if "symbol" in params:
        return 2
    else:
        return 1


# https://binance-docs.github.io/apidocs/futures/en/#symbol-order-book-ticker
ENDPOINT_SYMBOL_ORDER_BOOK_TICKER_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/ticker/bookTicker",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=weight_endpoint_symbol_order_book_ticker_futures
)

# https://binance-docs.github.io/apidocs/futures/en/#get-all-liquidation-orders
ENDPOINT_GET_ALL_LIQUIDATION_ORDERS_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/allForceOrders",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 5
)

# https://binance-docs.github.io/apidocs/futures/en/#open-interest
ENDPOINT_OPEN_INTEREST_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/openInterest",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#open-interest-statistics
ENDPOINT_OPEN_INTEREST_STATISTICS_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/futures/data/openInterestHist",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-accounts-market_data
ENDPOINT_TOP_TRADER_LONG_SHORT_RATIO_ACCOUNTS_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/futures/data/topLongShortAccountRatio",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#top-trader-long-short-ratio-positions
ENDPOINT_TOP_TRADER_LONG_SHORT_RATIO_POSITIONS_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/futures/data/topLongShortPositionRatio",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#taker-buy-sell-volume
ENDPOINT_TAKER_LONG_SHORT_BUY_SELL_VOLUME_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/futures/data/takerlongshortRatio",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#historical-blvt-nav-kline-candlestick
ENDPOINT_HISTORICAL_BLVT_CANDLESTICK_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v1/lvtKlines",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 1
)

# https://binance-docs.github.io/apidocs/futures/en/#account-information-v2-user_data
ENDPOINT_ACCOUNT_INFORMATION_FUTURES = Endpoint(
    base_endpoint=BASE_ENDPOINT_FUTURES,
    request_method=RequestMethod.GET,
    endpoint="/fapi/v2/account",
    endpoint_security_type=EndpointSecurityType.USER_DATA,
    weight_method=lambda x={}: 5
)


# ******************* Binance Spot Endpoints ********************** #

def weight_endpoint_candlestick_data_spot(params={}) -> int:
    if "limit" not in params:
        return 2
    assert (params["limit"] >= 1) and (params["limit"] <= 1000)
    return 2


# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data
ENDPOINT_CANDLESTICK_DATA_SPOT = Endpoint(
    base_endpoint=BASE_ENDPOINT_SPOT,
    request_method=RequestMethod.GET,
    endpoint="/api/v3/klines",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=weight_endpoint_candlestick_data_spot
)

# https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-endpoints#exchange-information
ENDPOINT_EXCHANGE_INFORMATION_SPOT = Endpoint(
    base_endpoint=BASE_ENDPOINT_SPOT,
    request_method=RequestMethod.GET,
    endpoint="/api/v3/exchangeInfo",
    endpoint_security_type=EndpointSecurityType.NONE,
    weight_method=lambda x={}: 20
)
