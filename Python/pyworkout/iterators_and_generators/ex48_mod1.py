import os
from typing import Generator


def files_reader(
    dir_path: str
) -> Generator[tuple, None, None]:
    for file_num, file_name in enumerate(os.listdir(dir_path), start=1):
        file_path = os.path.join(dir_path, file_name)
        try:
            for line_num, line in enumerate(open(file_path), start=1):
                yield (file_num, file_name, line_num, line)
        except OSError:
            pass


def main():
    dir_path = 'iterators_and_generators/'
    for idx, line in enumerate(files_reader(dir_path)):
        if idx == 10:
            break
        print(line)


if __name__ == '__main__':
    main()
