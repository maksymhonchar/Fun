#!/usr/bin/python3
# coding: utf-8

"""
Реализуйте функцию сжатия строки на основе счетчика повторяющихся символов.
Например, строка aabcccccaaa должна превратиться в а2b1с5аЗ.
Если «сжатая» строка оказывается длиннее исходной, метод должен вернуть исходную строку.
"""


from itertools import groupby


def compress_string(string) -> str:
    """Compress string to custom character-number format (see task2 description)
    Use itertools.groupby to create character groups
    Use str.join instead of string concatenation to speed up compression

    Arguments:
    string -- string to compress

    Returns:
    compressed_string if length of compressed string is less than or equal to length of compressed string
    string if length of compressed string is larger than length of uncompressed string

    Raises:
    TypeError if argument string is not a builtins.str or not any subclass of str
    """
    if not isinstance(string, str):
        raise TypeError
    if len(string) == 0:
        raise ValueError("Expected len(string) > 0, got 0")
    chars_groups = [
        list(group_items)
        for (_, group_items) in groupby(string)  # skip group_name
    ]
    compressed_chars_groups = [
        ''.join( [group_items[0], str(len(group_items))] )
        for group_items in chars_groups
    ]
    compressed_string = ''.join(compressed_chars_groups)
    string_result = string if len(compressed_string) > len(string) else compressed_string
    return string_result


def compress_string_nogroupby(string) -> str:
    """Compress string to custom character-number format (see task2 description)
    Iterate through array to find groups of characters
    Use str.join instead of string concatenation to speed up compression

    Arguments:
    string -- string to compress

    Returns:
    compressed_string if length of compressed string is less than or equal to length of compressed string
    string if length of compressed string is larger than length of uncompressed string

    Raises:
    TypeError if argument string is not a builtins.str or not any subclass of str
    """
    if not isinstance(string, str):
        raise TypeError
    if len(string) == 0:
        raise ValueError("Expected len(string) > 0, got 0")
    chars_groups = []  # [ [char, char_cnt] ]
    chars_groups.append([string[0], 1])
    for idx, char in enumerate(string[1:], 1):
        prev_char = chars_groups[-1][0]
        if char == prev_char:
            chars_groups[-1][1] += 1
        else:
            chars_groups.append([char, 1])
    compressed_chars_groups = [
        ''.join([group_items[0], str(group_items[1])])
        for group_items in chars_groups
    ]
    compressed_string = ''.join(compressed_chars_groups)
    string_result = string if len(compressed_string) > len(
        string) else compressed_string
    return string_result


def test_task2example():
    correct_result = 'a2b1c5a3'
    assert compress_string('aabcccccaaa') == correct_result, \
        'task2example 1: not passed'
    assert compress_string_nogroupby('aabcccccaaa') == correct_result, \
        'task2example 2: not passed'
    return True


def test_smallcompressedstr():
    correct_result = 'a2b1c5a3abcdefg'
    assert compress_string('a2b1c5a3abcdefg') == correct_result, \
        'smallcompressedstr 1: not passed'
    assert compress_string_nogroupby('a2b1c5a3abcdefg') == correct_result, \
        'smallcompressedstr 2: not passed'
    return True


def test_singlechar():
    correct_result = '13'
    assert compress_string('111') == correct_result, \
        'singlechar 1: not passed'
    assert compress_string_nogroupby('111') == correct_result, \
        'singlechar 2: not passed'
    return True


def test_unicode():
    correct_result = 'а2б1ц5а3'
    assert compress_string('а2б1ц5а3') == correct_result, \
        'unicode 1: not passed'
    assert compress_string_nogroupby('а2б1ц5а3') == correct_result, \
        'unicode 2: not passed'
    return True


if __name__ == "__main__":
    test_cases_funcs = [
        test_task2example, test_smallcompressedstr,
        test_singlechar,
        test_unicode
    ]
    test_cases_results = [func() for func in test_cases_funcs]
    print('All tests passed? - {0}'.format(all(test_cases_results)))
