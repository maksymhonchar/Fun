from decimal import Decimal, getcontext
from typing import Dict


def fahrenheit_to_celsius(
    temperature: float,
    precision: int = 5
) -> Decimal:
    getcontext().prec = precision
    return (Decimal(temperature) - Decimal(32.0)) / Decimal(1.8)


def transform_temperature(
    data: Dict[str, float]
) -> Dict[str, Decimal]:
    return {
        city: fahrenheit_to_celsius(temperature)
        for city, temperature in data.items()
    }


def main():
    data = {
        'kyiv': 70.5,
        'london': 35.1,
        'washington': 69.3
    }
    result = transform_temperature(data)
    print(result)


if __name__ == '__main__':
    main()
