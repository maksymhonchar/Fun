import re


def is_word(
    word: str
) -> bool:
    return word.isalnum() and any([char.isalpha() for char in word]) and word[0].isalpha()


def last_word(
    text: str
) -> str:
    tokens = re.split(pattern=r'(\b)', string=text)
    words = [word for word in tokens if is_word(word)]
    sorted_words = sorted(words, key=str.lower)
    last_word_alphabetically = sorted_words[-1]
    return last_word_alphabetically


def main() -> None:
    filepath = 'file.txt'
    with open(filepath, 'r') as fs_r:
        text = fs_r.read()
    print( last_word(text) )


if __name__ == '__main__':
    main()
