import datetime
import os
import pprint
from typing import Generator, Tuple


def file_usage_timing(
    dir_path: str
) -> Generator[Tuple[str, Tuple], None, None]:
    for file_name in os.listdir(dir_path):
        file_status = os.stat(os.path.join(dir_path, file_name))
        time_values = (
            datetime.datetime.fromtimestamp(file_status.st_atime),
            datetime.datetime.fromtimestamp(file_status.st_mtime),
            datetime.datetime.fromtimestamp(file_status.st_ctime)
        )
        yield file_name, time_values


def main():
    dir_path = 'functions/'
    for value in file_usage_timing(dir_path):
        pprint.pprint(value)


if __name__ == '__main__':
    main()
