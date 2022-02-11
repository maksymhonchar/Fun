def alternative_pig_latin(
    word: str
) -> str:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    unique_vowels = set()
    for character in set(word):
        if character in vowels:
            unique_vowels.add(character)

    word_has_enough_vowels = len(unique_vowels) >= 2
    if word_has_enough_vowels:
        return word + 'way'
    else:
        return word[1:] + word[0] + 'ay'


def main():
    word = input('Enter a word: ')
    print( alternative_pig_latin(word) )
    return encoded_word


if __name__ == '__main__':
    main()
