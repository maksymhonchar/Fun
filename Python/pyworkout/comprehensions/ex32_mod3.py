def analyze_config(
    file_path: str
) -> dict:
    separator = '='
    return {
        line.strip().split(separator)[0]: line.strip().split(separator)[1]
        for line in open(file_path)
    }


def main():
    file_path = 'comprehensions/file.txt'
    result = analyze_config(file_path)
    print(result)


if __name__ == '__main__':
    main()
