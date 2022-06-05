class Transaction:
    balance: float = 0.0

    def __init__(
        self,
        amount: float
    ) -> None:
        self.__class__.balance += amount

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {id(self)=} {self.balance=}'


def main():
    tx_1 = Transaction(100.0)
    tx_2 = Transaction(1540.0)
    tx_3 = Transaction(-1600.0)
    print(tx_1, tx_2, tx_3, end='\n---\n', sep='\n')


if __name__ == '__main__':
    main()
