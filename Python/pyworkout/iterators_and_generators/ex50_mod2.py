import os
from typing import Any, Generator


def my_chain(
    *iterables
) -> Generator[Any, None, None]:
    for iterable in iterables:
        for item in iterable:
            yield item


def all_lines(
    dir_path: str
) -> Generator:
    return my_chain(
        *(
            open(os.path.join(dir_path, file_name))
            for file_name in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, file_name))
        )
    )


def main():
    dir_path = 'functions/'
    for item in all_lines(dir_path):
        print(item)


if __name__ == '__main__':
    main()
