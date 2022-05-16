from decimal import Decimal

from freedonia.exceptions import HourIsOutOfRange, ProvinceIsUnknown


TAX_RATES = {
    "Chico": Decimal("0.5"),
    "Groucho": Decimal("0.7"),
    "Harpo": Decimal("0.5"),
    "Zeppo": Decimal("0.4")
}


def calculate_tax(
    amount: float,
    province: str,
    hour: int
) -> float:
    if province not in TAX_RATES:
        error_msg = f"Province argument value [{province}] is unknown"
        raise ProvinceIsUnknown(error_msg)

    if not (1 <= hour <= 24):
        error_msg = f"Hour argument value [{hour}] is out of range"
        raise HourIsOutOfRange(error_msg)

    tax = float(
        Decimal(amount) *
        TAX_RATES[province] *
        Decimal(hour / Decimal("24.0"))
    )
    final_price = amount + tax
    return final_price
