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
    operator_token, numbers_tokens = tokens[0], tokens[1:]

    operation = operator_table[operator_token]
    numbers = [float(number_token) for number_token in numbers_tokens]

    result = None
    for number, next_number in zip(numbers[0:], numbers[1:]):
        if result is None:
            result = operation(number, next_number)
        else:
            result = operation(result, next_number)

    return result


def main():
    expression = '+ 3 5 7'
    result = calc(expression)
    print(result)

    expression = '/ 100 5 5'
    result = calc(expression)
    print(result)


if __name__ == '__main__':
    main()
