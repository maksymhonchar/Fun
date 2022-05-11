from typing import List


def analyze_names(
    names: List[str]
) -> set:
    return {
        char
        for name in names
        for char in name
    }


def main():
    family_names = [
        'john',
        'max',
        'cameron'
    ]
    result = analyze_names(family_names)
    print(result)


if __name__ == '__main__':
    main()
