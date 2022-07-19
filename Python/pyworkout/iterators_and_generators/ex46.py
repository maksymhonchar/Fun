from typing import Any, Iterator, Sequence


class MyEnumerate:

    def __init__(
        self,
        data: Sequence,
        start: int = 0
    ) -> None:
        self._data = data
        self._index = start

    def __iter__(
        self
    ) -> Iterator:
        return self

    def __next__(
        self
    ) -> Any:
        if self._index < len(self._data):
            self._index += 1
            return self._index - 1, self._data[self._index - 1]
        else:
            raise StopIteration


def main():
    data = '12345'

    print('enumerate() ->')
    for idx, value in enumerate(data):
        print(f'{idx=} {value=}')

    print('-')

    print('MyEnumerate() ->')
    for idx, value in MyEnumerate(data):
        print(f'{idx=} {value=}')


if __name__ == '__main__':
    main()
