from freedonia import calculate_tax


def main():
    result = calculate_tax(100, 'Harpo', 12)
    print(result)

    result = calculate_tax(100, 'Harpo', 21)
    print(result)


if __name__ == '__main__':
    main()
