import pprint
from abc import ABC


class Animal(ABC):

    def __init__(
        self,
        color: str,
        number_of_legs: int,
        **kwargs
    ) -> None:
        self.color = color
        self.number_of_legs = number_of_legs
        if 'species' in kwargs:
            self.species = kwargs['species']
        else:
            self.species = self.__class__.__name__

    def __repr__(
        self
    ) -> str:
        return f'{self.color} {self.species}, {self.number_of_legs} legs'


class ZeroLeggedAnimal(Animal):

    def __init__(
        self,
        color: str,
        **kwargs
    ) -> None:
        number_of_legs = 0
        super().__init__(color, number_of_legs, **kwargs)


class Snake(ZeroLeggedAnimal):

    def __init__(
        self,
        color: str,
        **kwargs
    ) -> None:
        super().__init__(color, **kwargs)


def main():
    animals = [
        Snake('black', species='super_snake'),
    ]
    pprint.pprint(animals)


if __name__ == '__main__':
    main()
