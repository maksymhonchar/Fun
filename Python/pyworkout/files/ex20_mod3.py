import os
from collections import Counter, defaultdict


def analyze_dir_files(
    dir_path: str
) -> dict:
    analysis = defaultdict(int)

    try:
        for filename in os.listdir(dir_path):
            with open(os.path.join(dir_path, filename)) as fs_r:
                for line in fs_r:
                    for char, char_freq in Counter(line.lower()).items():
                        analysis[char] += char_freq
    except FileNotFoundError:
        error_msg = f'analyze_dir_files failed: cant find [{dir_path}]'
        raise FileNotFoundError(error_msg)

    return analysis


def main():
    dir_path = '.'
    analysis = analyze_dir_files(dir_path)
    print(analysis)


if __name__ == '__main__':
    main()
