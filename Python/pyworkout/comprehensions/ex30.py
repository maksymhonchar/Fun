from typing import List


def flatten(
    data: List[List]
) -> List:
    """Note: Works only for 2-level list"""
    return [
        item
        for list_item in data
        for item in list_item
    ]


def main():
    data = [[1, 2], [3, 4]]
    result = flatten(data)
    print(result)


if __name__ == '__main__':
    main()
