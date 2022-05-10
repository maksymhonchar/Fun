from typing import Callable


def transform_values_v2(
    map_func: Callable,
    filter_func: Callable,
    data: dict
) -> dict:
    return {
        key: map_func(value)
        for key, value in data.items()
        if filter_func(value)
    }


def main():
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = transform_values_v2(
        lambda x: x * x,
        lambda x: x % 2 == 0,
        data
    )
    print(result)


if __name__ == '__main__':
    main()
