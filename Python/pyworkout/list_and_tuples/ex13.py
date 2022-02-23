from operator import itemgetter
from typing import List as t_List


def print_people(
    people: t_List[tuple]
) -> None:
    lname_idx = 1
    fname_idx = 0
    for fname, lname, time in sorted(people, key=itemgetter(lname_idx, fname_idx)):
        print(f'{lname:<10} {fname:<10} {time:>5.2f}')


def main():
    people = [
        ('donald', 'trump', 7.8585173),
        ('joe', 'biden', 15.2898751),
        ('george', 'bush', 12.603)
    ]
    print_people(people)


if __name__ == '__main__':
    main()
