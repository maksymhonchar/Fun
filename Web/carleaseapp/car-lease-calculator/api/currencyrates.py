import requests


def get_nbu_exchange_rates() -> list:
    """
    Fetch exchage rates of UAH to many foreign currencies from NBU.

    Docs:
        https://bank.gov.ua/ua/open-data/api-dev

    Args:
        None

    Returns:
        list - list of dictionaries which describe exchange rates for different currencies
    """
    url = 'https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json'
    resp = requests.get(url=url)
    data = resp.json()
    return data


def get_currency_to_uah_nbu_exchange_rate(
    nbu_exchange_rates: list,
    currency_code: str
) -> dict:
    """Fetch exchange rate of UAH to specific foreign currency from NBU

    Note:
        in case NBU returned several blocks of information about single
        currency, method returns only single block of information.

    Docs:
        https://bank.gov.ua/ua/open-data/api-dev

    Args:
        nbu_exchange_rates: list - fetched list of currencies rates
        currency_code: str - code of the currency in upper case, according to ISO 4217

    Returns:
        dict - dictionary with exchange rate information for specific currency
        OR
        dict - empty dictionary in case NBU did not return data for requested currency
    """
    # Validate input
    if len(nbu_exchange_rates) == 0:
        raise ValueError('nbu_exchange_rates param is empty')

    # Find out exchange rate for aspecified currency
    specific_currency_information = [
        item
        for item in nbu_exchange_rates
        if item['cc'] == currency_code
    ]
    if len(specific_currency_information) >= 1:
        return specific_currency_information[0]
    else:
        return {}
