from menu import menu


def func_a():
    return "A"


def func_b():
    return "B"


def main():
    result = menu(a=func_a, b=func_b)
    print(f'{result=}')


if __name__ == '__main__':
    main()
