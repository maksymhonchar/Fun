import random
import string
from typing import Callable


def create_password_generator(
    chars: str
) -> Callable:
    def create(
        password_len: int
    ) -> str:
        return ''.join(
            random.choices(chars, k=password_len)
        )
    return create


def main():
    alpha_password = create_password_generator(string.ascii_letters)
    symbol_password = create_password_generator(string.punctuation)

    for password_len in (10, 15):
        alpha_result = alpha_password(password_len)
        print(f'{password_len=}; {alpha_result=}')

        symbol_result = symbol_password(password_len)
        print(f'{password_len=}; {symbol_result=}')


if __name__ == '__main__':
    main()
