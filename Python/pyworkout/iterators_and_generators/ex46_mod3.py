from typing import Any, Generator, Sequence


def my_enumerate_generator(
    data: Sequence
) -> Generator:
    return (
        (index, item)
        for index, item in zip(range(len(data)), data)
    )


def main():
    data = '12345'

    print('my_enumerate_generator() ->')
    for idx, value in my_enumerate_generator(data):
        print(f'{idx=} {value=}')

    my_enumerate_return_type = type(my_enumerate_generator(data))
    msg = f'my_enumerate_generator() return type is {my_enumerate_return_type}'
    print(msg)


if __name__ == '__main__':
    main()
