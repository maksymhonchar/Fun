from collections import Counter
from typing import List as t_List


def most_common_vowel_occurences(
    word: str
) -> int:
    vowels = {'a', 'o', 'i', 'e', 'u'}
    word_as_vowels = ''.join(
        [char for char in word if char.lower() in vowels ]
    )

    if len(word_as_vowels) == 0:
        return 0
    else:
        char, occurences = Counter(word_as_vowels).most_common(n=1)[0]
        return occurences


def word_with_most_vowels(
    words: t_List[str]
) -> str:
    word = max(words, key=most_common_vowel_occurences)
    return word



def main():
    strings = ['this', 'is', 'an', 'elementary', 'test', 'example', 'bbbb']
    result = word_with_most_vowels(strings)
    print(result)


if __name__ == '__main__':
    main()
