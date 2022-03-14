import csv

import numpy as np


def create_csv(
    filepath: str,
    lines_cnt: int
) -> None:
    line_integers_cnt = 10
    min_integer_value = 10
    max_integer_value = 100

    with open(filepath, 'w', newline='') as fs_w:
        writer = csv.writer(fs_w)
        for _ in range(lines_cnt):
            line_integers = np.random.randint(
                low=min_integer_value,
                high=max_integer_value,
                size=line_integers_cnt
            )
            writer.writerow(line_integers)


def analyze_csv(
    filepath: str
) -> list:
    analysis = []
    with open(filepath, 'r') as fs_r:
        reader = csv.reader(fs_r)
        for row in reader:
            if row:
                items = np.array(row, dtype=int)
                items_sum, items_mean = items.sum(), items.mean()
                value_to_save = (items_sum, items_mean)
                analysis.append(value_to_save)
    return analysis


def main():
    csv_filepath = 'file.csv'
    lines_cnt = 10
    create_csv(csv_filepath, lines_cnt)

    analysis = analyze_csv(csv_filepath)
    for idx, (items_sum, items_sum) in enumerate(analysis):
        print(f'Line {idx+1}: {items_sum=}, {items_sum=}')


if __name__ == '__main__':
    main()
