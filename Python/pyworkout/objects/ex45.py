from dataclasses import dataclass, field
from typing import List

from ex43 import Animal, Parrot, Sheep, Snake, Wolf
from ex44 import Cage


@dataclass
class Zoo:
    cages: List[Cage] = field(default_factory=list)

    def add_cages(
        self,
        *new_cages
    ) -> None:
        for new_cage in new_cages:
            self.cages.append(new_cage)

    def animals_by_color(
        self,
        color: str
    ) -> List[Animal]:
        return [
            animal
            for cage in self.cages
            for animal in cage.animals
            if animal.color == color
        ]

    def animals_by_legs(
        self,
        legs: int
    ) -> List[Animal]:
        return [
            animal
            for cage in self.cages
            for animal in cage.animals
            if animal.number_of_legs == legs
        ]

    def number_of_legs(
        self
    ) -> int:
        return sum([
            animal.number_of_legs
            for cage in self.cages
            for animal in cage.animals
        ])


def main():
    wolf = Wolf('black')
    sheep = Sheep('white')
    snake = Snake('white')
    parrot = Parrot('green')
    white_parrot = Parrot('white')

    c1 = Cage(1)
    c1.add_animals(wolf, sheep, white_parrot)

    c2 = Cage(2)
    c2.add_animals(snake, parrot)

    z = Zoo()
    z.add_cages(c1, c2)

    print(z)
    print(z.animals_by_color('white'))
    print(z.animals_by_legs(4))
    print(z.number_of_legs())


if __name__ == '__main__':
    main()
