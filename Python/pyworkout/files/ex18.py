def get_final_line(
    filepath: str
) -> str:
    with open(filepath) as fs_r:
        for line in fs_r:
            pass
        return line


def get_final_line__readlines(
    filepath: str
) -> str:
    with open(filepath) as fs_r:
        return fs_r.readlines()[-1]


def main():
    filepath = 'file.txt'
    final_line_content = get_final_line(filepath)
    print(f'Final line content: [{final_line_content}]')


if __name__ == '__main__':
    main()
