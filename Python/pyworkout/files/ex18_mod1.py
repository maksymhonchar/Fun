import re


def find_integers_sum(
    filepath: str
) -> int:
    integers_sum = 0

    with open(filepath) as fs_r:
        for line in fs_r:
            words = re.split('\W', line)
            integers = [int(word) for word in words if word.isdecimal()]
            integers_sum += sum(integers)

    return integers_sum


def main():
    filepath = 'file.txt'
    integers_sum = find_integers_sum(filepath)
    print(f'Integers sum is [{integers_sum}]')


if __name__ == '__main__':
    main()
