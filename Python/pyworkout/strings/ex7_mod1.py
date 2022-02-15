def ubbi_dubbi(
    word: str
) -> str:
    if len(word) == 0:
        return word

    vowels = {'a', 'e', 'i', 'o', 'u'}

    translated_word_letters = []
    for letter_idx, letter in enumerate(word):
        if letter.lower() in vowels:
            prefix_to_add = 'Ub' if (letter_idx == 0) and (letter.isupper()) else 'ub'
            translated_word_letters.append(prefix_to_add)
        translated_word_letters.append(letter.lower())

    translated_word = ''.join(translated_word_letters)
    
    return translated_word


def main():
    word = input('Enter a word to translate: ')
    print( ubbi_dubbi(word) )


if __name__ == '__main__':
    main()
