import urllib.request


def get_sv(
    data: str
) -> list:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return [
        word
        for word in data.split('\n')
        if all([vowel in word.lower() for vowel in vowels])
    ]


def get_sv_v2(
    data: str
) -> set:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return {
        word
        for word in data.split('\n')
        if set(word.lower()) & vowels == vowels
    }


def get_sv_v3(
    data: str
) -> set:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    return {
        word
        for word in data.split('\n')
        if vowels < set(word.lower())
    }


def main():
    data_url = r'https://gist.githubusercontent.com/reuven/9ea704169a2b633d8afd27fc340ad8c5/raw/7b255766069e4229a4a2498f429ecd01bb820f11/words.txt'
    data = urllib.request.urlopen(data_url).read().decode('utf-8')

    result = get_sv(data)
    print(result)


if __name__ == '__main__':
    main()
