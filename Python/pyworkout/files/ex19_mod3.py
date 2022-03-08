from pprint import pprint


def get_etcpasswd_users(
    filepath: str
) -> dict:
    users = {}
    separator = ':'
    comment_char = '#'

    with open(filepath) as fs_r:
        for line in fs_r:
            if not line.startswith((comment_char, '\n')):
                parsed_line = line.strip().split(separator)
                username = parsed_line[0]
                user = {
                    'uid': parsed_line[2],
                    'homedir': parsed_line[5],
                    'shell': parsed_line[6]
                }
                users[username] = user

    return users


def main():
    filepath = 'etcpasswd.txt'
    users = get_etcpasswd_users(filepath)
    pprint(users)


if __name__ == '__main__':
    main()
