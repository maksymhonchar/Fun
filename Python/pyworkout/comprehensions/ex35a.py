import pprint
from string import ascii_lowercase
from typing import Dict


def gematria_alphabet(
    first_letter: str = 'a',
    last_letter: str = 'z'
) -> Dict[str, int]:
    first_letter_idx = ord(first_letter)
    last_letter_idx = ord(last_letter)
    return {
        chr(letter_idx): letter_idx - first_letter_idx + 1
        for letter_idx in range(first_letter_idx, last_letter_idx + 1)
    }


def gematria_alphabet_v2() -> Dict[str, int]:
    return {
        letter: idx
        for idx, letter in enumerate(ascii_lowercase, start=1)
    }


def main():
    alphabet = gematria_alphabet()
    pprint.pprint(alphabet)


if __name__ == '__main__':
    main()
