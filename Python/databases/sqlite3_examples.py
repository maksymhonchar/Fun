import sqlite3

db_path = 'dbs/company.db'
connection = sqlite3.connect(db_path)  # "CREATE DATABASE company;"
cursor = connection.cursor()


def example_1():
    drop_table = \
        """
        DROP TABLE employee
        """
    cursor.execute(drop_table)

    create_employee = \
        """
        CREATE TABLE employee (
        staff_number INTEGER PRIMARY KEY,
        fname VARCHAR(20),
        lname VARCHAR(30),
        gender CHAR(1),
        joining DATE,
        birth_date DATE);
        """
    cursor.execute(create_employee)

    insert_employee = \
        """
        INSERT INTO employee (staff_number, fname, lname, gender, birth_date)
        VALUES (NULL, "max", "honchar", "m", "1998-08-07");
        """
    cursor.execute(insert_employee)

    insert_employee = \
        """
        INSERT INTO employee (staff_number, fname, lname, gender, birth_date)
        VALUES (NULL, "max2", "honchar2", "m", "1998-09-08");
        """
    cursor.execute(insert_employee)


def example_2():
    staff_data = [
        ("max3", "honchar3", "m", "1234-12-12"),
        ("max4", "honchar4", "m", "1234-12-12"),
    ]

    for entry in staff_data:
        format_str = \
            """
            INSERT INTO employee (staff_number, fname, lname, gender, joining, birth_date) 
            VALUES (NULL, "{first}", "{last}", "{gender}", NULL, "{birthdate}");
            """
        sql_insert_command = format_str.format(
            first=entry[0], last=entry[1], gender=entry[2], birthdate=entry[3]
        )
        cursor.execute(sql_insert_command)


def example_3():
    select_query = \
        """
        SELECT * FROM employee;
        """
    cursor.execute(select_query)
    # fetch all:
    print("fetch all:")
    result = cursor.fetchall()
    for entry in result:
        print(entry)
    # fetch one:
    cursor.execute(select_query)
    print("fetch one:")
    result = cursor.fetchone()
    print(result)


example_3()

# save the changes.
connection.commit()
connection.close()
