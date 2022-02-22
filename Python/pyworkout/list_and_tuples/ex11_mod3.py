from typing import List as t_List


def sort_data(
    data: t_List[list]
) -> t_List[list]:
    return sorted(
        data,
        key=lambda item: 0 if not item else sum(item)
    )


def main():
    data = [
        [],
        [1, 2, 3],
        [4.123, 0.0000123, -5]
    ]

    sorted_data = sort_data(data)

    print(
        '[data]:\n', data, '\n\n',
        '[sorted data]:\n', sorted_data,
        sep=''
    )


if __name__ == '__main__':
    main()
