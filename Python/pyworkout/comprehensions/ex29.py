def sum_numbers(
    content: str,
    separator: str = ' '
) -> int:
    return sum(
        [
            int(token)
            for token in content.split(separator)
            if token.isdecimal()
        ]
    )


def sum_numbers_using_mapfilter(
    content: str,
    separator: str = ' '
) -> int:
    return sum(
        map(
            int,
            filter(
                str.isnumeric,
                content.split(separator)
            )
        )
    )


def main():
    content = '10 abc 20 de44 30 55fg 40 50'

    result = sum_numbers(content)
    print(result)

    result = sum_numbers_using_mapfilter(content)
    print(result)


if __name__ == '__main__':
    main()
