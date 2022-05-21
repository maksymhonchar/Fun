import pprint
from typing import List


class Scoop(object):
    
    def __init__(
        self,
        flavor: str
    ) -> None:
        self.flavor = flavor

    def __repr__(
        self
    ) -> str:
        return f"{self.__class__} {id(self)=} {self.flavor=}"


def create_scoops(
    flavors: List[str]
) -> List[Scoop]:
    return [
        Scoop(flavor)
        for flavor in flavors
    ]


def main():
    flavors = ["chocolate", "vanilla", "persimmon"]
    scoops = create_scoops(flavors)
    pprint.pprint(scoops)


if __name__ == '__main__':
    main()
