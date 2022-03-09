from collections import defaultdict


def wordcount(
    filepath: str
) -> dict:
    stats = defaultdict(int)
    unique_words_storage = set()

    with open(filepath) as fs_r:
        for line in fs_r:
            words = [word for word in line.split()]
            stats['characters'] += len(line)
            stats['words'] += len(words)
            stats['lines'] += 1
            unique_words_storage.update(set(words))

    stats['unique_words'] += len(unique_words_storage)

    return stats


def main():
    filepath = 'file.txt'
    stats = wordcount(filepath)
    print(stats)


if __name__ == '__main__':
    main()
