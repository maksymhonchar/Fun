from collections import defaultdict


def get_stats(
    filepath: str
) -> dict:
    stats = defaultdict(list)
    separator = ':'
    comment_char = '#'

    with open(filepath) as fs_r:
        for line in fs_r:
            if not line.startswith((comment_char, '\n')):
                parsed_line = line.strip().split(separator)
                username, shell = parsed_line[0], parsed_line[6]
                stats[shell].append(username)

    return stats


def main():
    filepath = 'etcpasswd.txt'
    stats = get_stats(filepath)
    print(stats)


if __name__ == '__main__':
    main()
