import sys


def main() -> None:
    mylist = []
    max_size = 10
    for number in range(max_size):
        mylist.append(number)
        print(f'mylist len={len(mylist)}, size={sys.getsizeof(mylist)}')


if __name__ == '__main__':
    main()
