from typing import Any, Iterator, Sequence


class MyEnumerateIterator:

    def __init__(
        self,
        data: Sequence
    ) -> None:
        self._data = data
        self._index = 0

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


class MyEnumerate:

    def __init__(
        self,
        data: Sequence
    ) -> None:
        self._data = data

    def __iter__(
        self
    ) -> Iterator:
        return MyEnumerateIterator(self._data)


def main():
    e = MyEnumerate('abc')

    print('** A **')
    for index, one_item in e:
        print(f'{index}: {one_item}')

    print('** B **')
    for index, one_item in e:
        print(f'{index}: {one_item}')


if __name__ == '__main__':
    main()
