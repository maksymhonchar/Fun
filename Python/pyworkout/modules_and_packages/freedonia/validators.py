from freedonia.exceptions import HourIsOutOfRange, ProvinceIsUnknown
from freedonia.globals import TAX_RATES


def validate_calculate_tax(func):
    def wrapper(*args, **kwargs):
        _, province, hour = args
        if province not in TAX_RATES:
            error_msg = f"Province argument value [{province}] is unknown"
            raise ProvinceIsUnknown(error_msg)
        if not (1 <= hour <= 24):
            error_msg = f"Hour argument value [{hour}] is out of range"
            raise HourIsOutOfRange(error_msg)
        return func(*args, **kwargs)
    return wrapper
