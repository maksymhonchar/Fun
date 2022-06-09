from abc import ABC


class Animal(ABC):

    def __init__(
        self,
        color: str,
        number_of_legs: int
    ) -> None:
        self.color = color
        self.species = self.__class__.__name__
        self.number_of_legs = number_of_legs

    def __repr__(
        self
    ) -> str:
        return f'{self.color} {self.species}, {self.number_of_legs} legs'


class Sheep(Animal):

    def __init__(
        self,
        color: str,
        **kwargs
    ) -> None:
        number_of_legs = 4
        super().__init__(color, number_of_legs, **kwargs)

    def __repr__(
        self
    ) -> str:
        sound = 'BAA'
        return f'"{sound}" - {super().__repr__()}'


def main():
    s = Sheep('grey')
    print(s)


if __name__ == '__main__':
    main()
