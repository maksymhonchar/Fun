import pprint


def transform_etcpasswd(
    etcpasswd_file_path: str
) -> dict:
    separator = ':'
    username_field_idx = 0
    userid_field_idx = 3
    return {
        line.strip().split(separator)[username_field_idx]: \
            line.strip().split(separator)[userid_field_idx]
        for line in open(etcpasswd_file_path)
        if line.strip()
    }


def main():
    etcpasswd_file_path = 'comprehensions/etcpasswd.txt'
    result = transform_etcpasswd(etcpasswd_file_path)
    pprint.pprint(result)


if __name__ == '__main__':
    main()
