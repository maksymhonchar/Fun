def hex_to_dec(
    hex_digit: str
) -> int:
    ord_digit = ord(lower(hex_digit))
    if ord('0') <= ord_digit <= ord('9'):
        return ord_digit - ord('0')
    elif ord('a') <= ord_digit <= ord('f'):
        return ord_digit - ord('a') + 10
    else:
        raise ValueError('Got invalid character: expected hex 0-9, a-f')


def hex_output(
    hex_number_str: str
) -> int:
    hex_base = 16
    dec_number_int = 0

    if hex_number_str.startswith('-'):
        hex_number_str = hex_number_str.replace('-', '')
    if hex_number_str.startswith('0x'):
        hex_number_str = hex_number_str.replace('0x', '')

    for digit_idx, digit in enumerate(reversed(hex_number_str)):
        dec_number_int += hex_to_dec(digit) * (hex_base ** digit_idx)

    if hex_number_str.startswith('-'):
        dec_number_int *= -1

    return dec_number_int


def main():
    hex_number_str = input('Enter a number in hex: ')
    dec_number_int = hex_output(hex_number_str)
    print(f'Decimal value: {dec_number_int}')


if __name__ == '__main__':
    main()
