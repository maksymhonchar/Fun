def count_vowels(
    data: str
) -> dict:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return {
        token: sum([token.count(vowel) for vowel in vowels])
        for token in data.split()
    }


def main():
    data = 'this is an easy test'
    result = count_vowels(data)
    print(result)


if __name__ == '__main__':
    main()
