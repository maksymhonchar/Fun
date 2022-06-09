import pprint
from abc import ABC


class Animal(ABC):

    def __init__(
        self,
        color: str
    ) -> None:
        self.color = color
        self.species = self.__class__.__name__

    def __repr__(
        self
    ) -> str:
        return f'{self.color} {self.species}, {self.number_of_legs} legs'


class Sheep(Animal):
    number_of_legs = 4


class Snake(Animal):
    number_of_legs = 0


class Wolf(Animal):
    number_of_legs = 4


class Parrot(Animal):
    number_of_legs = 2


def main():
    animals = [
        Sheep('grey',),
        Snake('brown'),
        Wolf('black'),
        Parrot('green')
    ]
    pprint.pprint(animals)


if __name__ == '__main__':
    main()
