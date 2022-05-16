from typing import List
import pprint


def transform_data(
    data: List[tuple]
) -> dict:
    return {
        title: {
            'fname': name.split()[0],
            'lname': name.split()[1],
            'price': price
        }
        for name, title, price in data
    }


def main():
    sales_data = [
        ('first last 1', 'title 1', 30.5),
        ('first last 2', 'title 2', 16.2),
        ('first last 3', 'title 3', 96.6),
        ('first last 4', 'title 4', 15.9)
    ]
    result = transform_data(sales_data)
    pprint.pprint(result)


if __name__ == '__main__':
    main()
