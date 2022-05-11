def analyze_etcpasswd(
    etcpasswd_path: str
) -> set:
    separator = ':'
    command_shell_item_idx = -1
    return {
        line.strip().split(separator)[command_shell_item_idx]
        for line in open(etcpasswd_path)
    }


def main():
    etcpasswd_path = 'comprehensions/file.txt'
    result = analyze_etcpasswd(etcpasswd_path)
    print(result)


if __name__ == '__main__':
    main()
