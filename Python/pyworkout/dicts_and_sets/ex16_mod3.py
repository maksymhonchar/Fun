from typing import Callable as Callable_t


def dict_partition(
    d: dict,
    f: Callable_t
) -> tuple:
    d1, d2 = ({}, {})

    for key, value in d.items():
        if f(key, value):
            d1[key] = value
        else:
            d2[key] = value

    return d1, d2


def main():

    def func(
        item1: str,
        item2: str
    ) -> bool:
        if isinstance(item1, str) and (len(item1) > 4):
            if isinstance(item2, int) and (item2 > 10):
                return True
            else:
                return False
        else:
            return False

    dict_to_test = {
        'abcdefghi': 20,
        'abcdef': 5,
        'abc': 999,
    }
    result = dict_partition(dict_to_test, func)

    print(result)


if __name__ == '__main__':
    main()
