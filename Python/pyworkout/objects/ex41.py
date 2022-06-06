from dataclasses import dataclass
from dataclasses import field as dcls_field
from typing import ClassVar, List


@dataclass
class Scoop:
    flavor: str


@dataclass
class Bowl:
    scoops: List[Scoop] = dcls_field(default_factory=list)
    MAX_SCOOPS: ClassVar[int] = 3

    def add_scoops(
        self,
        *scoops
    ) -> None:
        for scoop in scoops:
            if len(self.scoops) < self.MAX_SCOOPS:
                self.scoops.append(scoop)


@dataclass
class BigBowl(Bowl):
    MAX_SCOOPS: ClassVar[int] = 5


def main():
    s1 = Scoop('brownies')
    s2 = Scoop('raspberry')
    s3 = Scoop('pear')
    s4 = Scoop('choco')
    s5 = Scoop('choco_v2')
    s6 = Scoop('choco_v3')

    bowl = Bowl()
    bowl.add_scoops(s1, s2, s3, s4, s5, s6)
    print(bowl)

    bigbowl = BigBowl()
    bigbowl.add_scoops(s1, s2, s3, s4, s5, s6)
    print(bigbowl)


if __name__ == '__main__':
    main()
