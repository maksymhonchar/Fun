from typing import Any, Generator, Tuple


def my_zip(
    *iterables
) -> Generator[Tuple, None, None]:
    min_iterable_len = min(iterables, key=len)
    for index in range(min_iterable_len):
        yield tuple(
            iterable[index]
            for iterable in iterables
        )


def main():
    print('builtins.zip:')
    for item in zip('abc', 'def', 'ghi'):
        print(item)

    print('---')

    print('ex50_mod1.my_zip:')
    for item in my_zip('abc', 'def', 'ghi'):
        print(item)


if __name__ == '__main__':
    main()
