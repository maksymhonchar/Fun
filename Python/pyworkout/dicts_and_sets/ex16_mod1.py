def dict_update(
    *dicts_to_add
) -> dict:
    result = {}

    for dict_to_add in dicts_to_add:
        result.update(dict_to_add)

    return result


def main():
    result = dict_update(
        {1: '1', 2: '2', 3: '3'},
        {3: '4', 4: 4, 5: 5},
        {'3': 3, 4: 5, 6: 6},
        {'3': '33'}
    )
    print(result)


if __name__ == '__main__':
    main()
