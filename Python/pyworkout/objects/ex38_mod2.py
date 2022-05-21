import pprint


class Beverage(object):

    def __init__(
        self,
        name: str,
        temperature: float = 75.0
    ) -> None:
        self.name = name
        self.temperature = temperature

    def __repr__(
        self
    ) -> str:
        return f"{self.__class__} {id(self)=} {self.name=} {self.temperature=}"


def main():
    beverages = [
        Beverage('bev1', 35.5),
        Beverage('bev3')
    ]
    pprint.pprint(beverages)


if __name__ == '__main__':
    main()
