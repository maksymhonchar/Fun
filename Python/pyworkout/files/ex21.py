import os
import pprint
import time
import re


def find_longest_word(
    filename: str
) -> str:
    try:
        with open(filename) as fs_r:
            file_longest_word = ''
            for line in fs_r:

                # Using for word in splitted line: t=0.007216799999999999
                for word in re.split('\W', line):
                    if len(word) > len(file_longest_word):
                        file_longest_word = word

                # Using sorted(): t=0.007218599999999999
                # sorted_words = sorted(re.split('\W', line), key=len)
                # if sorted_words:
                #     line_longest_word = sorted_words[-1]
                #     if len(line_longest_word) > len(file_longest_word):
                #         file_longest_word = line_longest_word

    except FileNotFoundError:
        error_msg = f'Cant find file [{filename}]'
        raise FileNotFoundError(error_msg)

    return file_longest_word


def find_all_longest_words(
    dir_path: str
) -> dict:
    return {
        filename: find_longest_word(os.path.join(dir_path, filename))
        for filename in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, filename))
    }


def time_decorator(func):
    def wrapper():
        t1 = time.perf_counter()
        func()
        t2 = time.perf_counter()
        print(f'Execution took [{t2 - t1}]s')
    return wrapper


@time_decorator
def main():
    dir_path = '.'
    all_longest_words = find_all_longest_words(dir_path)
    pprint.pprint(all_longest_words)


if __name__ == '__main__':
    main()
