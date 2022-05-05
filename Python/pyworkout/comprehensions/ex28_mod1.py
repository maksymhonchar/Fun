def join_numbers(
    integers: range,
    separator: str = ','
) -> str:
    return separator.join(
        [
            str(integer)
            for integer in integers
            if 0 <= integer <= 10
        ]
    )


def join_numbers_using_mapfilter(
    integers: range,
    separator: str = ','
) -> str:
    return separator.join(
        map(
            str,
            filter(
                lambda value: 0 <= value <= 10,
                integers
            )
        )
    )


def main():
    result = join_numbers(range(15))
    print(result)


if __name__ == '__main__':
    main()
