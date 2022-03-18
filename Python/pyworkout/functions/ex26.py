import operator
from numbers import Number


def calc(
    expression: str
) -> Number:
    operator_table = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
    }

    tokens = expression.split()

    token_is_number = [token.isnumeric() for token in tokens]
    if not any(token_is_number):
        error_msg = f'Cant find numeric values in {expression=}'
        raise ValueError(error_msg)

    first_number_token_idx = token_is_number.index(True)
    operators, numbers = tokens[:first_number_token_idx], tokens[first_number_token_idx:]
    if not operators:
        error_msg = f'Cant find operators in {expression=}'
        raise ValueError(error_msg)

    invalid_tokens_count = len(operators) != (len(numbers) - 1)
    if invalid_tokens_count:
        error_msg = f'Number of operators does not match number of numbers'
        raise ValueError(error_msg)

    result = None
    for operator_key, (number_l, number_r) in zip(operators, zip(numbers[0:], numbers[1:])):
        operation_func = operator_table[operator_key]
        if result is None:
            result = operation_func(float(number_l), float(number_r))
        else:
            result = operation_func(result, float(number_r))

    return result


def calc_v2(
    expression: str
) -> Number:
    operator_table = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
    }

    tokens = expression.split()

    result = None
    operation_idx = 0
    for token, next_token in zip(tokens[0:], tokens[1:]):
        if token not in operator_table:
            operation = operator_table[tokens[operation_idx]]
            operation_idx += 1
            if result is None:
                result = operation(float(token), float(next_token))
            else:
                result = operation(float(result), float(next_token))

    return result


def main():
    prefix_notation_expression = '* - + 5 7 5 123.123'

    result = calc(prefix_notation_expression)
    print(result)

    result = calc_v2(prefix_notation_expression)
    print(result)


if __name__ == '__main__':
    main()
