from collections import Counter
from typing import List


def get_stats(
    data: List[dict],
    n: int
) -> list:
    return Counter(
        [
            hobby
            for data_item in data
            for hobby in data_item['hobbies']
        ]
    ).most_common(n)


def main():
    data = [
        {
            'name': 'max',
            'hobbies': ['a', 'b', 'c']
        },
        {
            'name': 'max-2',
            'hobbies': ['d', 'a', 'a']
        },
        {
            'name': 'max-3',
            'hobbies': ['b', 'g', 'f']
        },
    ]
    stats = get_stats(data, n=3)
    print(stats)


if __name__ == '__main__':
    main()
