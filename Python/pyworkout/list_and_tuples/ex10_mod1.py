def apply_sum(a, b):
    if isinstance(a, dict):
        a_copy = a.copy()
        a_copy.update(b)
        return a_copy
    elif isinstance(a, set):
        return a | b
    else:
        return a + b


def mysum_bigger_than(
    threshold,
    *arguments
):
    if not arguments:
        raise ValueError('No arguments given')

    result = None
    for argument in arguments:
        if argument > threshold:
            if result is None:
                result = argument
            else:
                result = apply_sum(result, argument)

    return result


def main():
    print( mysum_bigger_than(10, 5, 20, 30, 6) )
    print( mysum_bigger_than([5], [1, 2, 3], [4, 5, 6], [7, 8, 9]) )
    print( mysum_bigger_than((5,), (1, 2, 3), (4, 5, 6), (7, 8, 9)) )
    print( mysum_bigger_than({5}, {1, 2, 3}, {4, 5, 6}, {7, 8, 9}) )  # weird


if __name__ == '__main__':
    main()
