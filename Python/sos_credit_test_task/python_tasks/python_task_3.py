#!/usr/bin/python3
# coding: utf-8

"""
Написать отдельный скрипт, принимающий из командной строки число, количество последних
строчек, которые нужно прочесть из файла и вывести в консоль.

Напишите предварительно отдельный скрипт, который генерирует текстовый файл и наполняет его
случайными числами. Количество строк передается из командной строки.
"""


import argparse
import os


def read_last_lines(filepath, lines_cnt) -> str:
    """Read last N lines from text file
    Assume the end of line is '\n' character

    Arguments:
    filepath -- full path to file to read
    lines_cnt -- number of last lines to read

    Returns:
    str -- last N lines from text file
    """
    if lines_cnt == 0:
        return ''
    if os.path.getsize(filepath) == 0:
        return ''
    with open(filepath, 'rb') as f:
        f.seek(-1, os.SEEK_END)  # jump to the last character
        if f.tell() == 0:  # if file consists of a single character
            f.seek(0, os.SEEK_SET)
            return f.readline().decode()
        for _ in range(lines_cnt):
            while f.read(1) != b'\n':
                try:
                    f.seek(-2, os.SEEK_CUR)
                except:
                    f.seek(0, os.SEEK_SET)
                    return ''.join([ line.decode() for line in f.readlines() ])
            f.seek(-2, os.SEEK_CUR)
        f.seek(2, os.SEEK_CUR)  # compensate -2 seek at the end of loop
        lines = ''.join([ line.decode() for line in f.readlines() ])
        return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read last N lines from text file')
    parser.add_argument("-l", help="number of last lines to read from file", type=int, required=True)
    parser.add_argument("-fp", help="full file path to text file", type=str, required=True)
    args = parser.parse_args()
    
    fullpath, linescnt = args.fp, args.l

    try:
        print( read_last_lines(fullpath, linescnt) )
    except IsADirectoryError:
        print('full path argument -fp is incorrect: point to file name, not a directory')
        parser.print_usage()
