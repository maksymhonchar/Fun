def sort_numbers(
    numbers: list
) -> list:
    return sorted(numbers, key=abs)


def main():
    numbers = [
        0,
        -1,
        1,
        2,
        -2,
        4,
        10,
        -999
    ]

    sorted_numbers = sort_numbers(numbers)

    print(
        '[numbers]:\n', numbers, '\n\n',
        '[sorted numbers]:\n', sorted_numbers, '\n\n',
        sep=''
    )


if __name__ == '__main__':
    main()
