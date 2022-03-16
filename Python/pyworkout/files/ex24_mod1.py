def encrypt(
    content: str
) -> str:
    encrypted_chars = [
        hex(ord(char))
        for char in content
    ]
    encrypted_str = ''.join(encrypted_chars)
    return encrypted_str


def decrypt(
    content: str
) -> str:
    hex_base = 16
    hex_tokens = [
        hex_token
        for hex_token in content.split('0x')[1:]
    ]
    decrypted_chars = [
        chr(int(hex_token, hex_base))
        for hex_token in hex_tokens
    ]
    decrypted_str = ''.join(decrypted_chars)
    return decrypted_str


def load_file(
    file_path: str
) -> str:
    with open(file_path) as fs_r:
        return fs_r.read()


def dump_file(
    file_path: str,
    content: str
) -> None:
    with open(file_path, 'w', newline='') as fs_w:
        fs_w.write(content)


def main():
    file_path = 'file.txt'
    file_content = load_file(file_path)

    encrypted_file_content = encrypt(file_content)
    print(f'{encrypted_file_content=}')

    dump_file(file_path, encrypted_file_content)

    file_content = load_file(file_path)
    decrypted_file_content = decrypt(file_content)
    print(f'{decrypted_file_content=}')


if __name__ == '__main__':
    main()
