import pprint


class Beverage(object):

    def __init__(
        self,
        name: str,
        temperature: float
    ) -> None:
        self.name = name
        self.temperature = temperature

    def __repr__(
        self
    ) -> str:
        return f"{self.__class__} {id(self)=} {self.name=} {self.temperature=}"


def main():
    names = ['bev1', 'bev2', 'bev3']
    temps = [38.5, 10.5, -3.2]
    beverages = [
        Beverage(name, temperature)
        for name, temperature in zip(names, temps)
    ]
    pprint.pprint(beverages)


if __name__ == '__main__':
    main()
