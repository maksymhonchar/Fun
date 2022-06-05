class Person:
    population: int = 0

    def __new__(cls):
        cls.population += 1
        print(f'DBG: new Person created, {cls.population=}')

        return object.__new__(cls)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {id(self)=} {self.population=}'


def main():
    people = [
        Person()
        for _ in range(5)
    ]
    print(people[0], people[-1], sep='\n')


if __name__ == '__main__':
    main()
