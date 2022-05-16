from typing import Dict, List
import pprint


def transform_data(
    user_currency: str,
    currencies_data: Dict[str, float],
    sales_data: List[tuple]
) -> Dict[str, dict]:
    unknown_currency = float("NaN")
    return {
        title: {
            'fname': full_name.split()[0].strip(),
            'lname': full_name.split()[1].strip(),
            'price': price_usd * currencies_data.get(user_currency, unknown_currency),
            'currency': user_currency
        }
        for full_name, title, price_usd in sales_data
    }


def main():
    currencies_data = {
        'UAH': 29.43,
        'EUR': 0.96,
        'JPY': 129.30
    }
    sales_data = [
        ('first last', 'title 1', 30.5),
        ('first last', 'title 2', 16.2),
        ('first last', 'title 3', 96.6),
        ('first last', 'title 4', 15.9)
    ]

    user_currency = 'UAH'
    result = transform_data(user_currency, currencies_data, sales_data)
    pprint.pprint(result)


if __name__ == '__main__':
    main()
