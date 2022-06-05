class Person:
    population: int = 0

    def __init__(self) -> None:
        self.__class__.population += 1

    def __del__(self) -> None:
        self.__class__.population -= 1
        del self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {id(self)=} {self.population=}'


def main():
    people = [
        Person()
        for _ in range(5)
    ]

    print(f'before: {people}')
    del people[0]

    print(f'after: {people}')


if __name__ == '__main__':
    main()
