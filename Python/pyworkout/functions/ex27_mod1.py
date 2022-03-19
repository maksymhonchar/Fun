import string
from typing import Callable


def create_password_checker(
    min_uppercase: int,
    min_lowercase: int,
    min_punctuation: int,
    min_digits: int
) -> Callable:
    def check(
        password: str
    ) -> bool:
        uppercase_count = sum([char.isupper() for char in password])
        if uppercase_count < min_uppercase:
            return False

        lowercase_count = sum([char.islower() for char in password])
        if lowercase_count < min_lowercase:
            return False

        punctuation_count = sum(
            [char in string.punctuation for char in password]
        )
        if punctuation_count < min_punctuation:
            return False

        digits_count = sum([char.isdigit() for char in password])
        if digits_count < min_digits:
            return False

        return True

    return check


def main():
    bad_password = 'hello'
    good_password = 'AA bb .. 12'

    checker = create_password_checker(2, 2, 2, 2)

    result = checker(bad_password)
    print(result)

    result = checker(good_password)
    print(result)


if __name__ == '__main__':
    main()
