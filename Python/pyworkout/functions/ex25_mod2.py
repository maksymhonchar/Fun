from numbers import Number

import numpy as np


def variable_multiply(
    *numbers
) -> Number:
    result = numbers[0]
    for number in numbers[1:]:
        result *= number
    return result


def recursive_multiply(
    *numbers
) -> Number:
    if len(numbers) == 1:
        return numbers[0]

    half = len(numbers) // 2
    part_1 = recursive_multiply(*numbers[:half])
    part_2 = recursive_multiply(*numbers[half:])
    return part_1 * part_2


def numpy_multiply(
    *numbers
) -> Number:
    return np.prod(numbers)


def main():
    result = variable_multiply(1, 2, 3, 1.1, 2.2, 3.3)
    print(f'Using variable: {result=}')

    result = recursive_multiply(1, 2, 3, 1.1, 2.2, 3.3)
    print(f'Using recursion: {result=}')

    result = numpy_multiply(1, 2, 3, 1.1, 2.2, 3.3)
    print(f'Using numpy.prod: {result=}')


if __name__ == '__main__':
    main()
