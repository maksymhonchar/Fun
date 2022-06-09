from collections.abc import Iterable
from typing import Any


class FlatList(list):

    def append(
        self,
        __object: Any
    ) -> None:
        if isinstance(__object, Iterable):
            for v in __object:
                super().append(v)
        else:
            super().append(__object)


def main():
    l = FlatList()
    l.append(123)
    l.append([1, 2, 3])
    print(l)


if __name__ == '__main__':
    main()
