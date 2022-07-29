from typing import Generator, Iterator


def my_range(
    end: int,
    step: int = 1
) -> Generator[int, None, None]:
    next_number = 0
    while next_number < end:
        yield next_number
        next_number += step


def main():
    print(f'{list(my_range(10))=}')


if __name__ == '__main__':
    main()
