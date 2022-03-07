def args_to_dict(
    *args
) -> dict:
    args_size = len(args)
    args_size_not_even = args_size % 2 != 0
    if args_size_not_even:
        error_msg = f'Expected even number of arguments. Got [{args_size}]'
        raise ValueError(error_msg)

    result = {}
    for key, value in zip(args[0::2], args[1::2]):
        result[key] = value

    return result


def main():
    result = args_to_dict('a', 1, 'b', 2, 'c', 3)
    print(result)


if __name__ == '__main__':
    main()
