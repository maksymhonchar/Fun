def mysum(*arguments):
    if not arguments:
        raise ValueError('No arguments given')

    first_elem_dtype = type(arguments[0])
    all_elements_share_same_dtype = all([
        isinstance(argument, first_elem_dtype)
        for argument in arguments
    ])
    if not all_elements_share_same_dtype:
        raise ValueError('Elements in arguments dont share same data type')

    result = arguments[0]
    for argument in arguments[1:]:
        if isinstance(argument, dict):
            result.update(argument)
        elif isinstance(argument, set):
            result |= argument
        else:
            result += argument

    return result


def main():
    # print()
    # print(mysum(1, 2, '3'))
    print(mysum(10))
    print(mysum(1, 2, 3))
    print(mysum(1.1, 2.2, 3.3))
    print(mysum('abc', 'def'))
    print(mysum([1, 2, 3], [4, 5, 6]))
    print(mysum({1, 2, 3}, {4, 5, 6}))
    print(mysum({1: 1}, {2: 2}))


if __name__ == '__main__':
    main()
