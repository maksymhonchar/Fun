from typing import List
from string import digits


def number_is_valid(
    number: str
) -> bool:
    separator = '-'
    area_code, telephone_prefix, line_number = \
        number.split(sep=separator, maxsplit=3)

    length_validators = [
        len(area_code) != 3,
        len(telephone_prefix) != 3,
        len(line_number) != 4
    ]
    if any(length_validators):
        return False

    number_validators = [
        not all([char in digits for char in number_part])
        for number_part in (area_code, telephone_prefix, line_number)
    ]
    if any(number_validators):
        return False

    return True


def transform_number(
    number: str
) -> str:
    area_code_last_char_idx = 3
    return '{0}{1}'.format(
        int(number[:area_code_last_char_idx]) + 1,
        number[area_code_last_char_idx:]
    )


def number_is_outdated(
    number: str
) -> bool:
    telephone_prefix_first_char_idx = 5
    allowed_chars = range(6)
    return int(number[telephone_prefix_first_char_idx]) in allowed_chars


def update_phone_numbers(
    numbers: List[str]
) -> List[str]:
    if any([not number_is_valid(number) for number in numbers]):
        error_msg = "Number format is invalid"
        raise ValueError(error_msg)

    return [
        transform_number(number) if number_is_outdated(number) else number
        for number in numbers
    ]


def main():
    numbers = ['123-456-7890', '123-333-4444', '123-777-8888']
    result = update_phone_numbers(numbers)
    print(result)


if __name__ == '__main__':
    main()
