from decimal import Decimal


def sum_float_vs_decimal() -> None:
    first_number_str = input('[A + B] Enter A: ')
    second_number_str = input('[A + B] Enter B: ')

    first_number_decimal = Decimal(first_number_str)
    second_number_decimal = Decimal(second_number_str)

    float_sum_result = float(first_number_str) + float(second_number_str)
    decimal_sum_result = first_number_decimal + second_number_decimal

    print(f'[A + B] using float: {float_sum_result}')
    print(f'[A + B] using Decimal: {decimal_sum_result}')


if __name__ == '__main__':
    sum_float_vs_decimal()
