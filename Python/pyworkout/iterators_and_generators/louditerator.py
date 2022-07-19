from typing import Any


class LoudIterator:

    def __init__(
        self,
        data
    ) -> None:
        print('__init__')
        self.data = data
        self.idx = 0

    def __iter__(
        self
    ):
        print('__iter__')
        return self

    def __next__(
        self
    ) -> Any:
        print('__next__')
        if self.idx < len(self.data):
            print('__next__ -> get value')
            to_return = self.data[self.idx]
            self.idx += 1
            return to_return
        else:
            print('__next__ -> raise StopIteration')
            raise StopIteration


def main():
    li = LoudIterator('012345')

    print('start loop')
    for item in li:
        print(f'Value inside the loop: {item}')


if __name__ == '__main__':
    main()
