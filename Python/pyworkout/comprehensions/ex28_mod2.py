from typing import List


def join_numbers(
    integers: List[str],
    separator: str = ','
) -> int:
    return sum(
        [
            int(hex_number, 16)
            for hex_number in integers
        ]
    )


def join_numbers_using_mapfilter(
    integers: List[str],
    separator: str = ','
) -> int:
    return sum(
        map(
            lambda value: int(value, 16),
            integers
        )
    )


def main():
    result = join_numbers_using_mapfilter(['0x10', '0x20', '0x30', '0x40'])
    print(result)


if __name__ == '__main__':
    main()
