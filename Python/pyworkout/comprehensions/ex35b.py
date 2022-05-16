from typing import Dict, List
from string import ascii_lowercase


def gematria_alphabet() -> Dict[str, int]:
    return {
        letter: idx
        for idx, letter in enumerate(ascii_lowercase, start=1)
    }


def gematria_for(
    alphabet: Dict[str, int],
    word: str
) -> int:
    unknown_char_score = 0
    return sum(
        [
            alphabet.get(char, unknown_char_score)
            for char in word
        ]
    )


def gematria_equal_words(
    alphabet: Dict[str, int],
    known_words: Dict[str, int],
    word: str
) -> List[str]:
    word_score = gematria_for(alphabet, word)
    return [
        known_word
        for known_word, score in known_words.items()
        if word_score == score
    ]


def main():
    alphabet = gematria_alphabet()

    word = 'cat'
    score = gematria_for(alphabet, word)
    print(score)


if __name__ == '__main__':
    main()
