def sum_dicts(
    *dict_arguments
) -> dict:
    result = {}

    for dict_argument in dict_arguments:
        for key, value in dict_argument.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    for key, value in result.items():
        if len(value) == 1:
            result[key] = value[0]

    return result


def main():
    print( sum_dicts({1:1}, {2:2}, {2:3}, {2:4}, {3:3}) )


if __name__ == '__main__':
    main()
