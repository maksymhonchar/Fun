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


class Bowl(object):

    def __init__(
        self
    ) -> None:
        self.scoops: List[Scoop] = []

    def add_scoops(
        self,
        *new_scoops
    ) -> None:
        for new_scoop in new_scoops:
            self.scoops.append(new_scoop)

    def __repr__(
        self
    ) -> str:
        return f"{self.__class__} {id(self)=} {self.scoops=}"


def main():
    s1 = Scoop('chocolate')
    s2 = Scoop('vanilla')
    s3 = Scoop('persimmon')

    b = Bowl()
    b.add_scoops(s1, s2)
    b.add_scoops(s3)

    print(b)


if __name__ == '__main__':
    main()
