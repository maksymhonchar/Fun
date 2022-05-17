from decimal import Decimal

from freedonia.globals import TAX_RATES
from freedonia.validators import validate_calculate_tax


@validate_calculate_tax
def calculate_tax(
    amount: float,
    province: str,
    hour: int
) -> float:
    tax = float(
        Decimal(amount) *
        Decimal(TAX_RATES[province]) *
        Decimal(Decimal(hour) / Decimal("24.0"))
    )
    final_price = amount + tax
    return final_price
