from collections import defaultdict


def get_vowels_stats(
    filepath: str
) -> dict:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    stats = defaultdict(int)

    with open(filepath) as fs_r:
        for line in fs_r:
            line_lower = line.lower()
            for vowel in vowels:
                stats[vowel] += line_lower.count(vowel)

    return stats


def display_stats(
    stats: dict
) -> None:
    for key, value in stats.items():
        print(f'{key}\t{value}')


def main():
    filepath = 'file.txt'
    stats = get_vowels_stats(filepath)
    display_stats(stats)


if __name__ == '__main__':
    main()
