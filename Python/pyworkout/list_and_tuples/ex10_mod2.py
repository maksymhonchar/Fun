def sum_numeric(
    *arguments
) -> int:
    result = 0

    for argument in arguments:
        try:
            result += int(argument)
        except:
            continue

    return result


def main():
    print( sum_numeric(10, 20, 'a', '30', 'bcd') )


if __name__ == '__main__':
    main()
