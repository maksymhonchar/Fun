#!/usr/bin/python3
# coding: utf-8

"""
Реализуйте функцию, определяющая, является ли одна строка перестановкой другой (палиндром).
Под перестановкой понимаем любое изменение порядка символов.
Регистр учитывается, пробелы являются существенными.
"""

"""
Комментарий кандидата: палиндром является частным случаем перестановки с любым изменением
порядка символов.
"""


from collections import Counter


def is_permutation(str_1, str_2) -> bool:
    """Check if str_1 and str_2 are permutations of each other.
    Use collections.Counter to calculate appearance frequency of each unique character.

    Arguments:
    str_1 -- first string
    str_2 -- other string

    Returns:
    True if str_1 is permutation of str_2
    False otherwise

    Raises:
    TypeError if one of arguments is not a builtins.str or not any subclass of str
    """
    if not isinstance(str_1, str) or not isinstance(str_2, str):
        raise TypeError
    if len(str_1) != len(str_2):
        return False
    is_permutation_result = (Counter(str_1) == Counter(str_2))
    return is_permutation_result


def is_permutation_nocounter(str_1, str_2) -> bool:
    """Check if str_1 and str_2 are permutations of each other
    Use str.count to calculate appearance frequency of each unique character

    Arguments:
    str_1 -- first string
    str_2 -- other string

    Returns:
    True if str_1 is permutation of str_2
    False otherwise

    Raises:
    TypeError if one of arguments is not a builtins.str or not any subclass of str
    """
    if not isinstance(str_1, str) or not isinstance(str_2, str):
        raise TypeError
    if len(str_1) != len(str_2):
        return False
    str_1_char_freq = { char: str_1.count(char) for char in set(str_1) }
    str_2_char_freq = { char: str_2.count(char) for char in set(str_2) }
    is_permutation_result = (str_1_char_freq == str_2_char_freq)
    return is_permutation_result


def test_palindrome():
    assert is_permutation('abcde', 'edcba'), \
        'palindrome 1: not passed'
    assert is_permutation_nocounter('abcde', 'edcba'), \
        'palindrome 2: not passed'
    return True


def test_permutation():
    assert is_permutation('SOS Credit', 'iCred tSSO'), \
        'permutation 1: not passed'
    assert is_permutation_nocounter('SOS Credit', 'iCred tSSO'), \
        'permutation 2: not passed'
    return True


def test_unicode():
    assert is_permutation('12е слово за словом', 'за словом 21е слово'), \
        'unicode 1: not passed'
    assert is_permutation_nocounter('12е слово за словом', 'за словом 21е слово'), \
        'unicode 2: not passed'
    return True


def test_wrongtype_counter():
    try:
        assert is_permutation(123, 321), 'wrongtype 1: not passed'
        raise AssertionError
    except TypeError:
        return True


def test_wrongtype_nocounter():
    try:
        assert is_permutation_nocounter(123, 321), 'wrongtype 2: not passed'
        raise AssertionError
    except TypeError:
        return True


if __name__ == "__main__":
    test_cases_funcs = [
        test_palindrome, test_permutation,
        test_unicode,
        test_wrongtype_counter, test_wrongtype_nocounter
    ]
    test_cases_results = [func() for func in test_cases_funcs]
    print('All tests passed? - {0}'.format( all(test_cases_results) ))
