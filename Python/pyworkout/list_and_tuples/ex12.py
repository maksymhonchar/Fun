from collections import Counter
from typing import List as t_List


def most_common_char_occurences(
    word: str
) -> int:
    char, occurences = Counter(word).most_common(n=1)[0]
    return occurences


def most_repeating_word_v1(
    words: t_List[str]
) -> str:
    word = max(words, key=most_common_char_occurences)
    return word


def most_repeating_word_v2(
    words: t_List[str]
) -> str:
    words_and_max_occurences = []

    for word in words:
        if len(word) == 0:
            max_occurences = 0
        else:
            char_occurences = [word.count(char) for char in set(word)]
            max_occurences = max(char_occurences)
        words_and_max_occurences.append((max_occurences, word))

    sorted_words = sorted(words_and_max_occurences, key=lambda item: item[0])
    occurences, word = sorted_words[-1]

    return word


def main():
    strings = ['this', 'is', 'an', 'elementary', 'test', 'example']
    result_v1 = most_repeating_word_v1(strings)
    result_v2 = most_repeating_word_v2(strings)
    print(result_v1, result_v2)


if __name__ == '__main__':
    main()
