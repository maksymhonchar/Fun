import pprint
from typing import Any


class RecentDict(dict):

    def __init__(
        self,
        max_items: int
    ) -> None:
        self.max_items = max_items

    def __setitem__(
        self,
        key: Any,
        value: Any
    ) -> None:
        if len(self) >= self.max_items:
            key_to_drop = next(iter(self))
            del self[key_to_drop]
        return super().__setitem__(key, value)


def main():
    d = RecentDict(5)
    for idx in range(10):
        d[idx] = f'{idx}_value'
    pprint.pprint(d)


if __name__ == '__main__':
    main()
