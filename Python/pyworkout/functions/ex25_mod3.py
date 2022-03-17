from typing import Any as Any_t
from typing import List as List_t


def anyjoin(
    values: List_t[Any_t],
    separator: str = ' '
) -> str:
    result = separator.join(
        [str(value) for value in values]
    )
    return result


def main():
    print(
        anyjoin([1, 2, 3])
    )
    print(
        anyjoin('abc', separator='**')
    )


if __name__ == '__main__':
    main()
