def name_triangle() -> None:
    name = input('Enter your name: ')

    for letter_idx, _ in enumerate(name):
        print( name[:letter_idx + 1] )


if __name__ == '__main__':
    name_triangle()
