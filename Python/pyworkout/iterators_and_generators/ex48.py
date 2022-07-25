import os
from typing import Generator


def files_reader(
    dir_path: str
) -> Generator:
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        try:
            for line in open(file_path):
                yield line
        except OSError:
            pass


def files_reader_using_walk(
    dir_path: str
) -> Generator:
    for _, _, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(dir_path, file_name)
            try:
                for line in open(file_path):
                    yield line
            except OSError:
                pass


def main():
    dir_path = 'functions/'
    for idx, line in enumerate(files_reader(dir_path)):
        if idx == 10:
            break
        print(line)


if __name__ == '__main__':
    main()
