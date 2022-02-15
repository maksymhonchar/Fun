import urllib.parse


def encode_url(
    url: str
) -> str:
    allowed_characters = {'/', '.', '-', '_', '~'}
    encoded_chars = []
    for char in url:
        if char.isalpha() or char.isdigit() or (char in allowed_characters):
            encoded_chars.append(char)
        else:
            char_as_hex = hex(ord(char)).upper()
            char_replacement = f'%{char_as_hex[2:]}'
            encoded_chars.append(char_replacement)
    encoded_url = ''.join(encoded_chars)
    return encoded_url


def encode_url_urlencode(
    url: str
) -> str:
    encoded_url = urllib.parse.quote(url)
    return encoded_url


def main():
    url = 'https://www.bbc.com/stories/hello world from the % crazy mega {town}!'

    encoded_url = encode_url(url)
    encoded_url_urlencode = encode_url_urlencode(url)

    print('Encoding URL: ')
    print(f'Before:\t\t[{url}]')
    print(f'After:\t\t[{encoded_url}]')
    print(f'urlencode:\t[{encoded_url_urlencode}]')


if __name__ == '__main__':
    main()
