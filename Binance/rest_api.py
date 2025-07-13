import aiohttp
import asyncio
import time
import hmac
from Binance.consts_enums import\
    API_KEY_READ_ONLY,\
    SECRET_KEY_READ_ONLY,\
    API_KEY_TRADING,\
    SECRET_KEY_TRADING,\
    API_KEY_WITHDRAWALS,\
    SECRET_KEY_WITHDRAWALS,\
    EndpointSecurityType,\
    Endpoint
from Binance.rate_limits import RequestWeightLimiterPerSecond


async def http_request(
        session: aiohttp.ClientSession,
        request_weight_limiter: RequestWeightLimiterPerSecond,
        endpoint: Endpoint,
        **kwargs
) -> tuple:
    """
    Makes a request to the endpoint.

    :param session:
    :param request_weight_limiter:
    :param endpoint:
    :param kwargs:
    :return: (status code, headers, data in json)
    """

    url = endpoint.base_endpoint + endpoint.endpoint
    session_request_method = getattr(session, endpoint.request_method.value)

    weight = endpoint.get_weight(kwargs.get("params", {}))

    # request is retried
    # in the case of defined status codes or exceptions
    retry_number = 5
    # (attempt ** retry_delay_power) seconds delay before retry
    retry_delay_power = 2

    retry_status = (
        500,
        501,
        502
    )

    # https://stackoverflow.com/questions/1434451/what-does-connection-reset-by-peer-mean/1434506#1434506
    retry_exception = (
        ConnectionResetError,
        asyncio.TimeoutError,
        aiohttp.ClientOSError,
        aiohttp.ClientResponseError,
        aiohttp.ServerDisconnectedError
    )

    # determines API Key to send in request headers
    headers = {}
    if endpoint.endpoint_security_type in (EndpointSecurityType.TRADE, EndpointSecurityType.MARGIN):
        headers = {"X-MBX-APIKEY": API_KEY_TRADING}
    elif endpoint.endpoint_security_type == EndpointSecurityType.WITHDRAWALS:
        headers = {"X-MBX-APIKEY": API_KEY_WITHDRAWALS}
    elif endpoint.endpoint_security_type != EndpointSecurityType.NONE:
        headers = {"X-MBX-APIKEY": API_KEY_READ_ONLY}

    # signs request
    # https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#signed-trade-and-user_data-endpoint-security
    if endpoint.endpoint_security_type in (
            EndpointSecurityType.TRADE,
            EndpointSecurityType.MARGIN,
            EndpointSecurityType.USER_DATA,
            EndpointSecurityType.WITHDRAWALS
    ):
        secret_key = SECRET_KEY_READ_ONLY
        if headers["X-MBX-APIKEY"] == API_KEY_TRADING:
            secret_key = SECRET_KEY_TRADING
        elif headers["X-MBX-APIKEY"] == API_KEY_WITHDRAWALS:
            secret_key = SECRET_KEY_WITHDRAWALS

        timestamp = time.time_ns() // 10**6
        recv_window = 5000

        params = kwargs.get("params", {})

        params["timestamp"] = timestamp
        params["recv_window"] = recv_window

        query_string = "&".join([f"{key}={value}" for key, value in params.items()])

        params["signature"] = hmac.new(secret_key.encode(), query_string.encode(), "sha256").hexdigest()

        kwargs["params"] = params

    attempt = 0
    while True:

        if request_weight_limiter.request_is_possible(weight):

            request_weight_limiter.add_weight(weight)

            try:

                async with session_request_method(url, headers=headers, **kwargs) as client_response:

                    if client_response.status != 429:

                        if (client_response.status in retry_status) and (attempt < retry_number):

                            attempt += 1
                            request_weight_limiter.subtract_weight(weight)
                            await asyncio.sleep(attempt ** retry_delay_power)
                            continue

                        client_response.raise_for_status()

                    request_weight_limiter.synchronize_used_weight(client_response)

                    return client_response.status, client_response.headers, await client_response.json()

            except retry_exception as exc:

                if attempt == retry_number:
                    raise exc
                attempt += 1
                request_weight_limiter.subtract_weight(weight)
                await asyncio.sleep(attempt ** retry_delay_power)

        else:

            await asyncio.sleep(1)


