import random
import string


def sort_words(
    words: list
) -> list:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return sorted(
        words,
        key=lambda word: sum([ 1 if char.lower() in vowels else 0 for char in word ])
    )


def main():
    words = []
    words_cnt = 10
    min_word_len = 3
    max_word_len = 10
    for _ in range(words_cnt):
        chars_cnt = random.randint(min_word_len, max_word_len)
        word = ''.join( random.choices(string.ascii_lowercase, k=chars_cnt) )
        words.append(word)

    sorted_words = sort_words(words)

    print(
        '[words]:\n', words, '\n\n',
        '[sorted words]:\n', sorted_words,
        sep=''
    )


if __name__ == '__main__':
    main()
