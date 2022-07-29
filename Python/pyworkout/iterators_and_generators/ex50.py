from itertools import chain
from typing import Any, Generator


def my_chain(
    *iterables
) -> Generator[Any, None, None]:
    for iterable in iterables:
        for item in iterable:
            yield item


def main():
    data = (
        'xyz',
        [1, 2, 3],
        {'a': 1, 'b': 2}
    )

    print('itertools.chain:')
    for item in chain(*data):
        print(item)

    print('---')

    print('ex50.my_chain:')
    for item in my_chain(*data):
        print(item)


if __name__ == '__main__':
    main()
