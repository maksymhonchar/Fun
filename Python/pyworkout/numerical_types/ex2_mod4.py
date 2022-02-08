def mysum(
    numbers: list,
    ignore: bool = True
) -> int:
    result = 0

    for number in numbers:
        number_is_int = str(number).isnumeric()
        if number_is_int:
            result += int(number)

    return result


if __name__ == '__main__':
    print( mysum([1, 2, 3, 4, 5.567, 'abc', None]) )
