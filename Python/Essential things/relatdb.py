# select insert delete rows.
stocks = [
    ('GOOG', 100, 490.1),
    ('AAPL', 50, 545.75),
    ('FB', 150, 7.45),
    ('HPQ', 75, 33.2)
]

import sqlite3

db = sqlite3.connect('database.db')
c = db.cursor()

def send_request():
    tmp_cursor = c.execute(
        'CREATE TABLE portfolio'
        '(symbol TEXT, shares INTEGER, price REAL)'
    )
    print(tmp_cursor, tmp_cursor.__class__)  # <class 'sqlite3.Cursor'>
    db.commit()

def insert_rows():
    c.executemany(
        'INSERT INTO portfolio VALUES (?,?,?)', stocks
    )
    db.commit()

def perform_select_query():
    for row in db.execute('SELECT * FROM portfolio'):
        print(row)

def perform_query_with_params(int_param):
    query = 'SELECT * FROM portfolio WHERE price <= ?'
    for row in db.execute(query, (int_param, )):
        print(row)


def main():
    perform_query_with_params(100)


if __name__ == '__main__':
    main()
