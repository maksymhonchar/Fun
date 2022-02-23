from collections import namedtuple
from operator import attrgetter
from typing import List as t_List


def print_people(
    people: t_List[tuple]
) -> None:
    for fname, lname, time in sorted(people, key=attrgetter('lname', 'fname')):
        print(f'{lname:<10} {fname:<10} {time:>5.2f}')


def main():
    people_record_nt = namedtuple('People', ['fname', 'lname', 'time'])
    people = [
        people_record_nt('donald', 'trump', 7.8585173),
        people_record_nt('joe', 'biden', 15.2898751),
        people_record_nt('george', 'bush', 12.603)
    ]
    print_people(people)


if __name__ == '__main__':
    main()
