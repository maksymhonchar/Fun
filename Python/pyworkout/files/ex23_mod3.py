import datetime
import json
import operator
import os


def get_dir_analysis(
    dir_path: str
) -> list:
    dir_analysis = []

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            file_stat = os.stat(file_path)
            last_modified = \
                datetime.datetime.utcfromtimestamp(
                    file_stat.st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")
            file_analysis = {
                'path': file_path,
                'size': file_stat.st_size,
                'last_modified': last_modified
            }
            dir_analysis.append(file_analysis)

    return dir_analysis


def dump_dir_analysis(
    file_path: str,
    dir_analysis: list
) -> None:
    with open(file_path, 'w') as fs_w:
        json.dump(dir_analysis, fs_w)


def display_dir_analysis(
    dir_analysis: list
) -> None:
    stats_sorted_by_last_modified = sorted(
        dir_analysis,
        key=operator.itemgetter('last_modified')
    )
    first_modified, last_modified = \
        stats_sorted_by_last_modified[0], stats_sorted_by_last_modified[-1]
    print(f'{first_modified=}, {last_modified=}')

    stats_sorted_by_size = sorted(
        dir_analysis,
        key=operator.itemgetter('size')
    )
    smallest, largest = \
        stats_sorted_by_size[0], stats_sorted_by_size[-1]
    print(f'{smallest=}, {largest=}')


def main():
    dir_path = input('Enter directory path: ')
    dir_analysis = get_dir_analysis(dir_path)

    dir_analysis_file_path = 'analysis.json'
    dump_dir_analysis(dir_analysis_file_path, dir_analysis)

    display_dir_analysis(dir_analysis)


if __name__ == '__main__':
    main()
