import os
from typing import Generator


def all_lines(
    dir_path: str,
    file_name_chunk: str
) -> Generator[str, None, None]:
    file_paths_to_read = [
        os.path.join(dir_path, file_name)
        for file_name in os.listdir(dir_path)
        if file_name_chunk in file_name
    ]
    for file_path in file_paths_to_read:
        for line in open(file_path):
            yield line


def main():
    dir_path = 'functions/'
    file_name_chunk = 'mod2'
    for line in all_lines(dir_path, file_name_chunk):
        print(line.rstrip())


if __name__ == '__main__':
    main()
