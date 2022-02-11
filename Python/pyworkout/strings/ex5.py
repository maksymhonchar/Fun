def pig_latin(
    word: str
) -> str:
    if len(word) == 0:
        raise ValueError('Enter a non-empty word')

    vowels = {'a', 'e', 'i', 'o', 'u'}

    first_letter_is_vowel = word[0].lower() in vowels
    if first_letter_is_vowel:
        return word + 'way'
    else:
        return word[1:] + word[0] + 'ay'


def main():
    word = input('Enter a word: ')
    print( pig_latin(word) )


if __name__ == '__main__':
    main()
