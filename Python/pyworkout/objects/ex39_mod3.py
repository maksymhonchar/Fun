from dataclasses import dataclass
from dataclasses import field as dcls_field
from typing import List


@dataclass
class Book():
    title: str
    author: str
    price: float
    width: float


@dataclass
class Shelf():
    width: float
    books: List[Book] = dcls_field(default_factory=list)

    def add_book(
        self,
        *books
    ) -> None:
        if sum([book.width for book in books]) > self.width:
            error_msg = "Books widths combined is larger than shelf width"
            raise ValueError(error_msg)

        for book in books:
            self.books.append(book)

    def total_price(
        self
    ) -> float:
        return sum([book.price for book in self.books])

    def has_book(
        self,
        title: str
    ) -> bool:
        return any([book.title == title for book in self.books])


def main():
    b1 = Book('b1', 'a1', 100.123, 25.0)
    b2 = Book('b2', 'a2', 200.123, 50.0)
    b3 = Book('b3', 'a3', 300.123, 10.0)

    shelf_width = 100.0
    shelf = Shelf(shelf_width)

    shelf.add_book(b1, b2, b3)


if __name__ == '__main__':
    main()
