from dataclasses import dataclass
from dataclasses import field as dcls_field
from typing import List


@dataclass
class Scoop:
    flavor: str


@dataclass
class Bowl:
    scoops: List[Scoop] = dcls_field(default_factory=list)
    MAX_SCOOPS: int = 3

    def add_scoop(
        self,
        *scoops
    ) -> None:
        for scoop in scoops:
            if len(self.scoops) < self.MAX_SCOOPS:
                self.scoops.append(scoop)


def main():
    s1 = Scoop('brownies')
    s2 = Scoop('raspberry')
    s3 = Scoop('pear')
    s4 = Scoop('choco')

    bowl = Bowl()
    bowl.add_scoop(s1, s2, s3, s4)

    print(bowl)


if __name__ == '__main__':
    main()
