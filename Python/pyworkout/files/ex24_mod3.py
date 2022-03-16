import csv
import os
from collections import defaultdict
from typing import DefaultDict as DefaultDict_t


def analyze_etcpasswd(
    etcpasswd_file_path: str
) -> DefaultDict_t[str, list]:
    etcpasswd_separator = ':'
    analysis = defaultdict(list)

    with open(etcpasswd_file_path) as fs_r:
        reader = csv.reader(fs_r, delimiter=etcpasswd_separator)
        for line_items in reader:
            if line_items:
                username, shell = line_items[0], line_items[-1],
                analysis[shell].append(username)

    return analysis


def write_analysis(
    analysis_file_path: str,
    analysis: DefaultDict_t[str, list]
) -> None:
    with open(analysis_file_path, mode='w', newline='') as fs_w:
        for shell, usernames in analysis.items():
            usernames_str = ', '.join(usernames)
            str_to_write = f'{shell}: {usernames_str}' + os.linesep
            fs_w.write(str_to_write)


def main():
    etcpasswd_file_path = 'etcpasswd'
    analysis = analyze_etcpasswd(etcpasswd_file_path)

    analysis_file_path = 'analysis.txt'
    write_analysis(analysis_file_path, analysis)


if __name__ == '__main__':
    main()
