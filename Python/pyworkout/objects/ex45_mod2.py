from dataclasses import dataclass, field
from typing import List, Union

from ex43 import Animal, Snake, Wolf
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

    def transfer_animal(
        self,
        target_zoo,
        animal_kind
    ) -> None:
        animal_to_transfer = self.extract_animal(animal_kind)
        if animal_to_transfer is None:
            error_msg = "Requested kind of animal is missing in the zoo"
            raise ValueError(error_msg)
        target_zoo.add_animal(animal_to_transfer)

    def extract_animal(
        self,
        kind
    ) -> Union[Animal, None]:
        for cage in self.cages:
            for idx, animal in enumerate(cage.animals):
                if isinstance(animal, kind):
                    return cage.animals.pop(idx)
        return None

    def add_animal(
        self,
        new_animal: Animal
    ) -> None:
        if not self.cages:
            error_msg = f"Cannot add {new_animal=} - missing cages"
            raise ValueError(error_msg)
        else:
            self.cages[0].add_animals(new_animal)


def main():
    wolf = Wolf('black')
    cage = Cage('zoo_1_cage_1')
    cage.add_animals(wolf)
    zoo = Zoo()
    zoo.add_cages(cage)

    snake = Snake('light-green')
    cage2 = Cage('zoo_2_cage_1')
    cage2.add_animals(snake)
    zoo2 = Zoo()
    zoo2.add_cages(cage2)

    print('Before:', zoo, zoo2, sep='\n')

    # zoo.transfer_animal(zoo2, Snake)
    zoo.transfer_animal(zoo2, Wolf)

    print('After:', zoo, zoo2, sep='\n')


if __name__ == '__main__':
    main()
