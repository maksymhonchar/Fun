def flip_dict(
    data: dict
) -> dict:
    return {
        value: key
        for key, value in data.items()
    }


def main():
    data = {'a': 1, 'b': 2, 'c': 3}
    result = flip_dict(data)
    print(result)


if __name__ == '__main__':
    main()
