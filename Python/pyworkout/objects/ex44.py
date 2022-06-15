from dataclasses import dataclass, field
from typing import List

from ex43 import Animal, Parrot, Sheep, Snake, Wolf


@dataclass
class Cage:
    identificator: str
    animals: List[Animal] = field(default_factory=list)

    def add_animals(
        self,
        *new_animals
    ) -> None:
        for animal in new_animals:
            self.animals.append(animal)


def main():
    sheep = Sheep('grey')
    snake = Snake('brown')
    wolf = Wolf('black')
    parrot = Parrot('green', species='super_parrot')

    c1 = Cage('1_31236')
    c1.add_animals(wolf, sheep)
    print(c1)

    c2 = Cage('2_5961')
    c2.add_animals(snake, parrot)
    print(c2)


if __name__ == '__main__':
    main()
