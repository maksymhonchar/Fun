def join_numbers(
    integers: range,
    separator: str = ','
) -> str:
    return separator.join(
        [
            str(integer)
            for integer in integers
        ]
    )


def main():
    result = join_numbers(range(15))
    print(result)


if __name__ == '__main__':
    main()
