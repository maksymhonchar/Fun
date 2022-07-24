from typing import Any, Iterator, Sequence


class CircleIterator:

    def __init__(
        self,
        sequence: Sequence[Any],
        max_iterations: int
    ) -> None:
        self.sequence = sequence
        self.max_iterations = max_iterations
        self.cur_iteration = 0
        self._sequence_len = len(self.sequence)

    def __iter__(
        self
    ) -> Iterator:
        return self

    def __next__(
        self
    ) -> Any:
        if self._sequence_len == 0:
            raise StopIteration
        if self.max_iterations <= 0:
            raise StopIteration
        if self.cur_iteration >= self.max_iterations:
            raise StopIteration

        self.cur_iteration += 1
        return self.sequence[(self.cur_iteration - 1) % self._sequence_len]


class Circle(CircleIterator):

    def __init__(
        self,
        sequence: Sequence[Any],
        max_iterations: int
    ) -> None:
        super().__init__(sequence, max_iterations)


def main():
    circle_obj = Circle([1, 2, 3, 4], 6)
    for item in circle_obj:
        print(item)


if __name__ == '__main__':
    main()
