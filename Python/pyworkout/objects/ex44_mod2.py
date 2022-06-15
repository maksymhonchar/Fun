from dataclasses import dataclass, field
from uuid import UUID, uuid4

from ex43 import Parrot


class SizedParrot(Parrot):

    def __init__(
        self,
        color,
        space_required,
        **kwargs
    ) -> None:
        self.space_required = space_required
        super().__init__(color, **kwargs)

    def __repr__(
        self
    ) -> str:
        parrot_repr = super().__repr__()
        return f'{parrot_repr}, {self.space_required} m3 required'


@dataclass
class SizedCage:
    max_size: float
    identificator: UUID = field(default_factory=lambda: uuid4())
    animals: list = field(default_factory=list)

    def add_animal(
        self,
        *new_animals
    ) -> None:
        for new_animal in new_animals:
            existing_animals_size = sum(
                animal.space_required
                for animal in self.animals
            )
            if (existing_animals_size + new_animal.space_required) <= self.max_size:
                self.animals.append(new_animal)
            else:
                error_msg = 'Not enough space to fit new animal'
                raise ValueError(error_msg)


def main():
    parrot = SizedParrot('grey', 32.5)
    cage = SizedCage(88.1643)

    cage.add_animal(parrot)
    cage.add_animal(parrot)
    cage.add_animal(parrot)
    print(cage)


if __name__ == '__main__':
    main()
