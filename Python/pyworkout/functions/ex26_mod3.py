from typing import Callable


def transform_lines(
    func: Callable,
    input_file_path: str,
    output_file_path: str
) -> None:
    with open(input_file_path) as input_fs, open(output_file_path, mode='w', newline='') as output_fs:
        for line in input_fs:
            line_to_write = func(line)
            output_fs.write(line_to_write)


def main():
    input_file_path = 'in_file.txt'
    output_file_path = 'out_file.txt'
    transform_lines(
        lambda line: line.upper(),
        input_file_path,
        output_file_path
    )


if __name__ == '__main__':
    main()
