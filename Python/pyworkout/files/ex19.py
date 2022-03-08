def etcpasswd_to_dict(
    filepath: str
) -> dict:
    stats = {}
    separator = ':'
    comment_char = '#'

    with open(filepath) as fs_r:
        for line in fs_r:
            if not line.startswith((comment_char, '\n')):
                parsed_line = line.strip().split(separator)
                username, uid = parsed_line[0], parsed_line[2]
                stats[username] = uid

    return stats


def main():
    filepath = 'etcpasswd.txt'
    stats = etcpasswd_to_dict(filepath)
    print(stats)


if __name__ == '__main__':
    main()
