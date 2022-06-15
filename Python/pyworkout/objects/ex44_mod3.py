from typing import Dict, List

from ex43 import Animal, Parrot, Sheep, Snake, Wolf


class Cage:

    def __init__(
        self,
        compatibility: Dict[Animal, List[Animal]]
    ) -> None:
        self.compatibility = compatibility
        self.animals: List[Animal] = []

    def add_animals(
        self,
        *new_animals
    ) -> None:
        for new_animal in new_animals:
            safe_cohabitants = self.compatibility[new_animal.__class__]
            all_cohabitants_are_safe = all([
                animal.__class__ in safe_cohabitants
                for animal in self.animals
            ])
            if all_cohabitants_are_safe:
                self.animals.append(new_animal)

    def __repr__(
        self
    ) -> str:
        return f'{self.__class__.__name__} {id(self)=} {self.animals=}'


def main():
    compatibility = {
        Parrot: [Parrot, Sheep],
        Sheep: [Sheep, Parrot],
        Wolf: [],
        Snake: [Snake]
    }
    cage = Cage(compatibility)

    sheep = Sheep('grey')
    snake = Snake('brown')
    wolf = Wolf('black')
    parrot = Parrot('green', species='super_parrot')

    cage.add_animals(sheep, sheep, sheep, parrot, wolf, snake)
    print(cage)


if __name__ == '__main__':
    main()
