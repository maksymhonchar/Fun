from typing import List


def flatten_odd_ints(
    data: List[List]
) -> List:
    """Note: Works only for 2-level list"""
    return [
        item
        for list_item in data
        for item in list_item
        if (isinstance(item, int) and (item % 2 == 1))
    ]


def main():
    data = [[1, 2], [3, 4], ['hello', 5, '7']]
    result = flatten_odd_ints(data)
    print(result)


if __name__ == '__main__':
    main()
