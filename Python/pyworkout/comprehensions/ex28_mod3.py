from typing import List


def reverse_file_content(
    file_path: str
) -> List[str]:
    with open(file_path) as fs_r:
        separator = ' '
        return [
            separator.join(
                line.strip().split(separator)[::-1]
            )
            for line in fs_r
        ]


def main():
    file_path = 'comprehensions/file.txt'
    result = reverse_file_content(file_path)
    print(result)


if __name__ == '__main__':
    main()
