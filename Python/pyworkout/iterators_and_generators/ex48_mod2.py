import os
from typing import Generator


def files_reader(
    dir_path: str
) -> Generator[str, None, None]:
    file_objs = [
        open(os.path.join(dir_path, file_name))
        for file_name in os.listdir(dir_path)
    ]
    while True:
        try:
            for file_obj in file_objs:
                yield file_obj.__next__()
        except StopIteration:
            break


def main():
    dir_path = 'iterators_and_generators/tmp/'
    for line in files_reader(dir_path):
        print(line.strip())


if __name__ == '__main__':
    main()
