from typing import Callable


def transform_values(
    func: Callable,
    data: dict
) -> dict:
    return {
        key: func(value)
        for key, value in data.items()
    }


def main():
    data = {'a': 1, 'b': 2, 'c': 3}
    result = transform_values(
        lambda x: x*x,
        data
    )
    print(result)


if __name__ == '__main__':
    main()
