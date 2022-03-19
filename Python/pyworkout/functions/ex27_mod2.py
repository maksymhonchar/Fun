from typing import Any, Callable


def getitem(
    from_obj: Any
) -> Callable:
    def get(
        key: Any
    ) -> Any:
        return from_obj[key]
    return get


def main():
    from_obj = {1: 1, 2: 2, 3: 3}
    itemgetter = getitem(from_obj)
    item = itemgetter(1)
    print(item)


if __name__ == '__main__':
    main()
