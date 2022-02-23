from operator import itemgetter
from typing import List as t_List


def display_data(
    data: t_List[tuple],
    columns: t_List[str]
) -> None:
    print('{0:<10}\t{1:<10}\t{2:>5}'.format(*columns))
    print('{0}\t{1}\t{2}'.format('-'*10, '-'*10, '-'*5))
    for record in data:
        print('{0:<10}\t{1:<10}\t{2:>5.2f}'.format(*record))


def main():
    data = [
        ('donald', 'trump', 7.8585173),
        ('joe', 'biden', 15.2898751),
        ('george', 'bush', 12.603)
    ]
    columns = ['name', 'length', 'time']

    while True:
        print('Existing data: ')
        display_data(data, columns)

        sort_column = input('Choose column to sort: ')
        if sort_column.lower() not in columns:
            print('Column does not exist')
        else:
            idx_to_sort = columns.index(sort_column)
            data = sorted(data, key=itemgetter(idx_to_sort))

        separator = '\n###############################\n'
        print(separator)


if __name__ == '__main__':
    main()
