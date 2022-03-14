import csv


def except_filenotfound(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except FileNotFoundError:
            error_msg = f'{func.__name__} failed: catched FileNotFoundError'
            raise FileNotFoundError(error_msg)
    return wrapper


@except_filenotfound
def passwd_to_csv(
    input_filepath: str,
    output_filepath: str
) -> None:
    reader_valid_line_items = 7

    input_file_params = {
        'file': input_filepath,
        'mode': 'r',
        'encoding': 'utf-8'
    }
    csv_reader_params = {
        'delimiter': ':'
    }
    output_file_params = {
        'file': output_filepath,
        'mode': 'w',
        'newline': '',
        'encoding': 'utf-8'
    }
    csv_writer_params = {
        'delimiter': '\t'
    }

    with open(**input_file_params) as fs_r, open(**output_file_params) as fs_w:
        csv_reader = csv.reader(fs_r, **csv_reader_params)
        csv_writer = csv.writer(fs_w, **csv_writer_params)
        for row in csv_reader:
            if len(row) == reader_valid_line_items:
                uname, uid = row[0], row[2]
                csv_writer.writerow( (uname, uid) )


def main():
    input_filepath = 'etcpasswd'
    output_filepath = './output.tsv'
    passwd_to_csv(input_filepath, output_filepath)


if __name__ == '__main__':
    main()
