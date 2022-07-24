from typing import Iterator

from multimethod import overload


class MyRangeIterator:

    def __init__(
        self,
        start: int,
        stop: int,
        step: int
    ) -> None:
        self.start = start
        self.stop = stop
        self.step = step
        self.next_number = self.start

    def __iter__(
        self
    ) -> Iterator:
        return self

    def __next__(
        self
    ) -> int:
        if abs(self.next_number) >= abs(self.stop):
            raise StopIteration
        self.next_number += self.step
        return self.next_number - self.step


class MyRange:

    @overload
    def __init__(
        self,
        stop: int
    ) -> None:
        self.start = 0
        self.stop = stop
        self.step = 1

    @overload
    def __init__(
        self,
        start: int,
        stop: int,
        step: int = 1
    ) -> None:
        self.start = start
        self.stop = stop
        self._validate_step(step)
        self.step = step

    def __iter__(
        self
    ) -> Iterator:
        return MyRangeIterator(self.start, self.stop, self.step)

    @staticmethod
    def _validate_step(
        value: int
    ) -> None:
        if value == 0:
            msg = 'MyRange() arg 3 (step) must not be zero'
            raise ValueError(msg)



def main():
    print(f'MyRange(10): {[value for value in MyRange(10)]}')
    print(f'MyRange(10, 20, 3): {[value for value in MyRange(10, 20, 3)]}')
    print(f'MyRange(10, -20, -3): {[value for value in MyRange(10, -20, -3)]}')


if __name__ == '__main__':
    main()
