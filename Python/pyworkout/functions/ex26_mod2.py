from typing import Callable, Iterable


def apply_to_each(
    func: Callable,
    values: Iterable
) -> list:
    return list(map(func, values))


def apply_to_each_v2(
    func: Callable,
    values: Iterable
) -> list:
    return [func(value) for value in values]


def main():
    values = [1, 2, 3, 4, 5]
    result = apply_to_each(
        lambda value: value + 100,
        values
    )
    print(f'Before: {values=}. After: {result=}')


if __name__ == '__main__':
    main()
