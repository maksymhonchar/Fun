import os


def write_to_file(
    dest_file_path: str,
    content: str
) -> None:
    with open(dest_file_path, mode='w', newline='') as dest_file_fs:
        dest_file_fs.write(content + os.linesep)


def copyfile(
    src_file_path: str,
    *dest_file_paths
) -> None:
    with open(src_file_path) as src_file_fs:
        for line in src_file_fs:
            for dest_file_path in dest_file_paths:
                write_to_file(dest_file_path, line)


def main():
    src_file_path = 'myfile.txt'
    copyfile(src_file_path, 'copy1.txt', 'copy2.txt', 'copy3.txt')


if __name__ == '__main__':
    main()
