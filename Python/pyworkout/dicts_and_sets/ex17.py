def how_many_different_numbers(
    numbers: list
) -> int:
    return len(set(numbers))


def main():
    numbers = [1, 2, 3, 1, 2, 3, 4, 1]
    result = how_many_different_numbers(numbers)
    print(result)


if __name__ == '__main__':
    main()
