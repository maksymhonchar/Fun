def mysum(
    numbers: list,
    start: int = 0
) -> int:
    result = start

    for number in numbers:
        result += number

    return result


if __name__ == '__main__':
    print(mysum([1, 2, 3, 4, 5.567]))
    print(mysum([1, 2, 3, 4, 5.567], 1))
