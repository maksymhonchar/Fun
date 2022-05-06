from typing import List


def get_specific_lines(
    file_path: str
) -> List[str]:
    vowels = ['a', 'e', 'i', 'o', 'u']
    with open(file_path) as fs_r:
        return [
            line
            for line in fs_r
            if (len(line) >= 20) and
            any([vowel in line.lower() for vowel in vowels])
        ]


def main():
    file_path = 'comprehensions/file.txt'
    result = get_specific_lines(file_path)
    print(result)


if __name__ == '__main__':
    main()
