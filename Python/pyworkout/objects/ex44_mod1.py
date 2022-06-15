from dataclasses import dataclass, field
from typing import List
from uuid import UUID, uuid4

from ex43 import Animal, Parrot, Sheep, Snake, Wolf


@dataclass
class BigCage:
    max_animals: int
    identificator: UUID = field(
        default_factory=lambda: uuid4()
    )
    animals: List[Animal] = field(default_factory=list)

    def add_animals(
        self,
        *new_animals
    ) -> None:
        for animal in new_animals:
            if len(self.animals) < self.max_animals:
                self.animals.append(animal)


def main():
    sheep = Sheep('grey')
    snake = Snake('brown')
    wolf = Wolf('black')
    parrot = Parrot('green', species='super_parrot')

    c1 = BigCage(1)
    c1.add_animals(snake)
    print(c1)

    c2 = BigCage(2)
    c2.add_animals(wolf, parrot, sheep)
    print(c2)


if __name__ == '__main__':
    main()
