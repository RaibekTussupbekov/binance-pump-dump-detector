from Binance.consts_enums import DATETIME_FORMAT
from datetime import datetime, timezone
import aiohttp


class RequestWeightLimiterPerMinute:
    """
    https://binance-docs.github.io/apidocs/spot/en/#limits
    https://binance-docs.github.io/apidocs/futures/en/#limits

    Manages minute request weight rate.

    According to the exchange information,
    base endpoint has only one request weight limiter with the following parameters:
    {
        "rateLimitType": "REQUEST_WEIGHT",
        "interval": "MINUTE",
        "intervalNum": 1,
        "limit": 1200 (for Spot) or 2400 (for Futures)
    }

    Use baseline:

        request_weight_limiter = RequestWeightLimiterPerMinute(2400)
        if request_weight_limiter.request_is_possible(weight):
            ...
        else:
            asyncio.sleep(1)
    """

    def __init__(self, limit: int):

        # The limits on the API are based on the IPs, not the API keys
        # Set weight limit per 1 minute
        self._limit = limit

        # weight used at the current minute
        self._used_weight = 0

        self._current_minute = datetime.now(timezone.utc).minute

    def _update_used_weight(self):
        if self._current_minute != datetime.now(timezone.utc).minute:
            self._current_minute = datetime.now(timezone.utc).minute
            self._used_weight = 0

    def add_weight(self, weight: int):
        self._used_weight += weight

    def subtract_weight(self, weight: int):
        if self._current_minute == datetime.now(timezone.utc).minute:
            self._used_weight -= weight

    def request_is_possible(self, weight: int) -> bool:
        self._update_used_weight()
        return weight <= (self._limit - self._used_weight)

    def synchronize_used_weight(self, client_response: aiohttp.ClientResponse):
        """
        Must be invoked only with either 200 or 429 status code after each request

        :param client_response: HTTP response on the request to the Binance endpoint
        :return: None
        """

        # HTTP 429 return code is used when breaking a request rate limit.
        # When a 429 is received, it's your obligation as an API to back off and not spam the API.
        if client_response.status == 429:

            self._current_minute = datetime.now(timezone.utc).minute
            self._used_weight = self._limit
            print(f"{datetime.now()}: 429 status code means that the limit is broken!")

        else:

            client_response_date = datetime.strptime(client_response.headers['Date'], DATETIME_FORMAT)
            client_response_used_weight = int(client_response.headers['X-MBX-USED-WEIGHT-1M'])
            self._update_used_weight()
            if client_response_date.minute == self._current_minute:
                self._used_weight = max(self._used_weight, client_response_used_weight)


class RequestWeightLimiterPerSecond:
    """
    https://binance-docs.github.io/apidocs/spot/en/#limits
    https://binance-docs.github.io/apidocs/futures/en/#limits

    Manages minute request weight rate at one second intervals.

    Although we already have RequestWeightLimiterPerMinute in our disposal,
    this limit manager for the second intervals was developed because
    the experience showed that Binance endpoints tend to add some extra weight
    to the requests if they're bombarded by a client which, in turn, leads to 429 status code.

    According to the exchange information,
    base endpoint has only one request weight limiter with the following parameters:
    {
        "rateLimitType": "REQUEST_WEIGHT",
        "interval": "MINUTE",
        "intervalNum": 1,
        "limit": 1200 (for Spot) or 2400 (for Futures)
    }

    Use baseline:

        request_weight_limiter = RequestWeightLimiterPerSecond(2400//60)
        if request_weight_limiter.request_is_possible(weight):
            ...
        else:
            asyncio.sleep(1)

    """

    def __init__(self, limit: int):

        # The limits on the API are based on the IPs, not the API keys
        # Set weight limit per 1 second
        self._limit = limit

        # weight used at the current second
        self._used_weight = 0

        self._current_second = datetime.now(timezone.utc).second

    def _update_used_weight(self):
        if self._current_second != datetime.now(timezone.utc).second:
            self._current_second = datetime.now(timezone.utc).second
            self._used_weight = 0

    def add_weight(self, weight: int):
        self._used_weight += weight

    def subtract_weight(self, weight: int):
        if self._current_second == datetime.now(timezone.utc).second:
            self._used_weight -= weight

    def request_is_possible(self, weight: int) -> bool:
        self._update_used_weight()
        return weight <= (self._limit - self._used_weight)

    def synchronize_used_weight(self, client_response: aiohttp.ClientResponse):
        """
        Must be invoked only with either 200 or 429 status code after each request

        :param client_response: HTTP response on the request to the Binance endpoint
        :return: None
        """

        # HTTP 429 return code is used when breaking a request rate limit.
        # When a 429 is received, it's your obligation as an API to back off and not spam the API.
        if client_response.status == 429:

            self._current_second = datetime.now(timezone.utc).second
            self._used_weight = self._limit
            print(f"{datetime.now()}: 429 status code means that the limit is broken!")

        else:

            client_response_date = datetime.strptime(client_response.headers['Date'], DATETIME_FORMAT)
            client_response_used_weight = int(client_response.headers['X-MBX-USED-WEIGHT-1M'])
            self._update_used_weight()
            if client_response_date.second == self._current_second:
                self._used_weight = max(self._used_weight, client_response_used_weight % (client_response_date.second+1))
