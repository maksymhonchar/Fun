def mysum(
    numbers: list
) -> int:
    result = 0

    for number in numbers:
        result += number

    return result


def mymean(
    numbers: list
) -> float:
    numbers_is_empty = len(numbers) == 0
    if numbers_is_empty:
        return float('nan')

    sum_value = mysum(numbers)
    mean_value = sum_value / len(numbers)

    return mean_value


if __name__ == '__main__':
    print( mymean([1, 2, 3, 4, 5,]) )
