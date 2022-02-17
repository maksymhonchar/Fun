def indices_sum(
    numbers
) -> tuple:
    even_indexed_numbers = sum(numbers[::2])
    odd_indexed_numbers = sum(numbers[1::2])
    return even_indexed_numbers, odd_indexed_numbers


def main() -> None:
    print( indices_sum( numbers=[10, 20, 30, 40, 50, 60] ) )


if __name__ == '__main__':
    main()
