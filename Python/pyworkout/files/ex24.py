import os


def transform_file(
    from_file_path: str,
    to_file_path: str
) -> None:
    from_open_params = {
        'file': from_file_path,
        'mode': 'r'
    }
    to_open_params = {
        'file': to_file_path,
        'mode': 'w',
        'newline': ''
    }

    with open(**from_open_params) as fs_r, open(**to_open_params) as fs_w:
        for line in fs_r:
            line_to_write = line.strip()[::-1] + os.linesep
            fs_w.write(line_to_write)


def main():
    from_file_path = 'from.txt'
    to_file_path = 'to.txt'
    transform_file(from_file_path, to_file_path)


if __name__ == '__main__':
    main()
