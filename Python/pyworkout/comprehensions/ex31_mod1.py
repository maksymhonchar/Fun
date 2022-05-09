from typing import Callable


def plword(
    word: str
) -> str:
    if len(word) == 0:
        error_msg = "can't process an empty word"
        raise ValueError(error_msg)

    vowels = {'a', 'e', 'i', 'o', 'u'}

    first_letter_is_vowel = word[0].lower() in vowels
    if first_letter_is_vowel:
        return word + 'way'
    else:
        return word[1:] + word[0] + 'ay'


def funcfile(
    file_path: str,
    function: Callable
) -> str:
    with open(file_path) as fs_r:
        return ' '.join(
            function(word)
            for line in fs_r
            for word in line.split()
        )


def main():
    file_path = 'comprehensions/file.txt'
    result = funcfile(file_path, plword)
    print(result)


if __name__ == '__main__':
    main()
