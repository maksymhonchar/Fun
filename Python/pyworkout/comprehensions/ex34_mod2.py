def word_lengths(
    file_path: str
) -> set:
    return {
        len(token)
        for line in open(file_path)
        for token in line.split()
    }


def main():
    file_path = 'comprehensions/file.txt'
    result = word_lengths(file_path)
    print(result)


if __name__ == '__main__':
    main()
