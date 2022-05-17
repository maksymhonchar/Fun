from typing import Callable


def dict_keys(
    data: dict,
    func: Callable
) -> dict:
    return {
        key: func(key)
        for key in data
    }


def main():
    data = {"a": 1, "b": 2, "c": 3, }
    func = lambda value: value * 3
    result = dict_keys(data, func)
    print(f"[{data=}] [{result=}]")


if __name__ == "__main__":
    main()
