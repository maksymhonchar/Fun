import string


def pig_latin(
    word: str
) -> str:
    vowels = {'a', 'e', 'i', 'o', 'u'}

    encoded_word = word

    last_letter_is_punctuation = word[-1] in string.punctuation
    if last_letter_is_punctuation:
        encoded_word = word[:-1]

    first_letter_is_vowel = encoded_word[0].lower() in vowels
    if first_letter_is_vowel:
        encoded_word = encoded_word + 'way'
    else:
        encoded_word = encoded_word[1:] + encoded_word[0] + 'ay'

    if last_letter_is_punctuation:
        encoded_word = encoded_word + word[-1]

    return encoded_word


def main():
    word = input('Enter a word: ')
    print( pig_latin(word) )


if __name__ == '__main__':
    main()
