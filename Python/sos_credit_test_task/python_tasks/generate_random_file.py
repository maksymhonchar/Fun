#!/usr/bin/python3
# coding: utf-8


import argparse
import os
import random
import string


def generate_random_file(filepath, lines_cnt):
    """Create new file filled with random text

    Arguments:
    filepath -- path to new text file
    lines_cnt -- number of lines in new text file
    """
    if lines_cnt < 0:
        lines_cnt = 0

    random.seed()
    LINE_MAX_LEN = 100
    with open(filepath, 'w') as file:
        for i in range(lines_cnt):
            random_line_tokens = random.choices(string.digits, k=LINE_MAX_LEN)
            random_line_tokens.append(os.linesep)
            random_line = ''.join(random_line_tokens)
            file.write(random_line)
        file.write('eof')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create random text file')
    parser.add_argument("-l", help="number of lines in random text file", type=int, required=True)
    parser.add_argument("-fp", help="full file path to random text file", type=str, default='./random_text.txt')
    args = parser.parse_args()
    
    fullpath, linescnt = args.fp, args.l

    try:
        generate_random_file(fullpath, linescnt)
    except IsADirectoryError:
        print('full path argument -fp is incorrect: point to file name, not a directory')
        parser.print_usage()
