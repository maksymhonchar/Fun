from typing import Any, Callable


def doboth(
    f1: Callable,
    f2: Callable
) -> Callable:
    def doboth_func(
        arg: Any
    ) -> Any:
        return f2(f1(arg))
    return doboth_func


def main():
    def f1(value): return value + 1
    def f2(value): return value + 10
    doboth_obj = doboth(f1, f2)

    arg = 100
    result = doboth_obj(arg)
    print(result)


if __name__ == '__main__':
    main()
