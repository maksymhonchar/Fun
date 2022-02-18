import requests


class NBUHandler(object):

    def get_exchange_rates(self) -> list:
        """Get UAH exchage rates from NBU

        Docs:
            https://bank.gov.ua/ua/open-data/api-dev
            "1. Офіційний курс гривні до іноземних валют та облікова ціна банківських металів"

        Args:
            None

        Returns:
            list - list of dictionaries which describe exchange rates for different currencies
        """
        url = 'https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?json'
        resp = requests.get(url=url)
        data = resp.json()
        return data

    def get_uah_exchange_rate(
        self,
        exchange_rates: list,
        currency_code: str
    ) -> list:
        """Get UAH exchange rate to specific currency from NBU

        Args:
            nbu_exchange_rates: list - exchange rates from NBU
            currency_code: str - code of the currency in upper case, according to ISO 4217

        Returns:
            list - list with exchange rate information for specific currency
            OR
            list - empty list in case exchange rate information for specific currency is missing
        """
        # Validate input
        if len(exchange_rates) == 0:
            raise ValueError('exchange_rates param is empty')

        # Get UAH exchange rate to specific currency
        specific_currency_information = [
            item
            for item in exchange_rates
            if item['cc'] == currency_code
        ]
        return specific_currency_information
