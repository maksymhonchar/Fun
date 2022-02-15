def pig_latin(
    word: str
) -> str:
    if len(word) == 0:
        return word

    vowels = {'a', 'e', 'i', 'o', 'u'}

    first_letter_is_vowel = word[0].lower() in vowels
    if first_letter_is_vowel:
        return word + 'way'
    else:
        return word[1:] + word[0] + 'ay'


def pig_latin_sentence(
    sentence: str
) -> str:
    separator = ' '

    encoded_words = [
        pig_latin(word)
        for word in sentence.split(sep=separator)
    ]

    encoded_sentence = separator.join(encoded_words)

    return encoded_sentence


def main():
    sentence = input('Enter a sentence: ')
    encoded_sentence = pig_latin_sentence(sentence)
    print(f'Result: [{encoded_sentence}]')


if __name__ == '__main__':
    main()
