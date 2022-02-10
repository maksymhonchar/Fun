def cut_float_using_str(
    number: float,
    before: int,
    after: int
) -> float:
    if (before < 0) or (after < 0):
        error_msg = 'Either [before] or [after] argument is invalid. Positive value is expected'
        raise ValueError(error_msg)

    number_as_str = str(number)
    number_as_str = number_as_str.replace('-', '')
    integer_part_str, fractional_part_str = number_as_str.split('.')

    if (before > len(integer_part_str)) or (after > len(fractional_part_str)):
        error_msg = 'Either [before] or [after] argument is invalid. One of them is out of bounds'
        raise ValueError(error_msg)

    truncated_integer_part_str = integer_part_str[-before:]
    truncated_fractional_part_str = fractional_part_str[:after]

    result_str = f'{truncated_integer_part_str}.{truncated_fractional_part_str}'
    if number < 0:
        result_str = '-' + result_str

    result_float = float(result_str)

    return result_float


def cut_float_using_math(
    number: float,
    before: int,
    after: int
) -> float:
    truncated_integer_part = int( number - (number // 10**before) * (10**before) )
    truncated_fractional_part = int( (number - int(number)) * (10**after) ) / (10**after)

    result_float = truncated_integer_part + truncated_fractional_part

    return result_float


if __name__ == '__main__':
    print( cut_float_using_str(12345.678901, 3, 3) )
    print( cut_float_using_math(12345.678901, 2, 2) )
