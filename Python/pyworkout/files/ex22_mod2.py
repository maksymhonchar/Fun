import csv


def dump_data(
    data: dict,
    filepath: str,
    delimiter: str
) -> None:
    with open(filepath, 'w', newline='') as fs_w:
        writer = csv.writer(fs_w, delimiter=delimiter)
        for key, value in data.items():
            values_to_write = (
                key,
                repr(value),
                type(value)
            )
            writer.writerow(values_to_write)


def main():
    data = {
        'one': 1,
        'two': 2,
        'three': ['drei', 3, 3.0]
    }
    filepath = 'file.tsv'
    delimiter = '\t'
    dump_data(data, filepath, delimiter)


if __name__ == '__main__':
    main()
