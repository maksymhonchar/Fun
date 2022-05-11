def read_config(
    config_file_path: str
) -> dict:
    separator = '='
    key_idx = 0
    value_idx = 1
    return {
        line.strip().split(separator)[key_idx]:
            int(line.strip().split(separator)[value_idx])
        for line in open(config_file_path)
        if line
        and line.strip().split(separator)[value_idx].isdecimal()
    }


def main():
    config_file_path = 'comprehensions/config.txt'
    result = read_config(config_file_path)
    print(result)


if __name__ == '__main__':
    main()
