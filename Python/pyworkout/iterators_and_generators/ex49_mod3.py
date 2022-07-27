from typing import Any, Callable, Generator, Iterable


def customized_apply(
    data: Iterable[Any],
    func: Callable
) -> Generator[Any, None, None]:
    print('[generator] starting for loop')
    for item in data:
        print(f'[generator] loop: {item=}')
        if func(item):
            yield item


def main():
    data = [1, 2, 3, 4, 5]
    func = lambda value: value % 2 == 0
    for item in customized_apply(data, func):
        print(f'[main] {item=}')


if __name__ == '__main__':
    main()
