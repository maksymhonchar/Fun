def pig_latin(
    word: str
) -> str:
    if len(word) == 0:
        raise ValueError('Enter a non-empty word')

    vowels = {'a', 'e', 'i', 'o', 'u'}

    first_letter_is_vowel = word[0].lower() in vowels
    if first_letter_is_vowel:
        encoded_word = word + 'way'
    else:
        encoded_word = word[1:] + word[0] + 'ay'

    first_letter_is_capitalized = word[0].isupper()
    if first_letter_is_capitalized:
        encoded_word = encoded_word.lower()
        encoded_word = encoded_word[0].upper() + encoded_word[1:]

    return encoded_word


def main():
    word = input('Enter a word: ')
    print( pig_latin(word) )


if __name__ == '__main__':
    main()
