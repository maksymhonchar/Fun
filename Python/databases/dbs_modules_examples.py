"""Just to retain some modules."""

def ADO_DB_API():
    """Microsoft ADO/SQL Server connections. Note that module has not been updated since 2014"""
    import adodbapi

    database = "db1.mdb"
    constr = 'Provider=Microsoft.Jet.OLEDB.4.0; Data Source=%s' % database
    tablename = "address"

    conn = adodbapi.connect(constr)
    cur = conn.cursor()

    try:
        sql = "select * from %s" % tablename
        cur.execute(sql)

        result = cur.fetchall()
        for item in result:
            print(item)
    finally:
        cur.close()
        conn.close()


def PY_ODBC():
    """Standart API for accessing databases. Crossplatformed: Win+Linux."""
    import pyodbc
    driver = 'DRIVER={SQL Server}'
    server = 'SERVER=localhost'
    port = 'PORT=1433'
    db = 'DATABASE=testdb'
    user = 'UID=me'
    pw = 'PWD=pass'

    conn_str = ';'.join([driver, server, port, db, user, pw])

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    cursor.execute('select * from table')

    row = cursor.fetchone()
    rest_of_rows = cursor.fetchall()


def MYSQL():
    import MySQLdb

    conn = MySQLdb.connect('localhost', 'username', 'pass', 'table_name')
    cursor = conn.cursor()

    cursor.execute('select * from table_name')

    row = cursor.fetchone()

    conn.close()


def PostgreSQL():
    import psycopg2

    conn = psycopg2.connect(
        dbname='mydb', user='me'
    )
    cursor = conn.cursor()

    cursor.execute('select * from table_name')
    row = cursor.fetchone()

    cursor.close()
    conn.close()


def SQLObject():
    from sqlobject import sqlhub, connectionForURI, SQLObject, StringCol, IntCol

    sqlhub.processConnection = connectionForURI('sqlite:/:memory:')

    class Person(SQLObject):
        name = StringCol()
        gender = StringCol(length=1, default=None)
        age = IntCol()

    Person.createTable()

    p = Person(name='Max', gender='m', age=19)
    p.name = 'Max2'
    print(p)

    p2 = Person.get(1)  # get_by_id
    print(p2)

    print(p is p2)  # True (!)

    # way to select 1
    p3 = Person.selectBy(name='Max2')[0]
    print(p3)

    # way to select 2
    p4 = Person.select(Person.q.gender == 'm')  # query
    print(p4[0], p4.count())  # 1


def PEEWEE():
    from peewee import SqliteDatabase, Model, CharField, DateField, BooleanField, ForeignKeyField

    db = SqliteDatabase('people.db')

    class Person(Model):
        name = CharField()
        birthday = DateField()
        gender = BooleanField()

        class Meta:
            database = db

    class Pet(Model):
        owner = ForeignKeyField(Person, related_name='pets')
        name = CharField()
        type = CharField()

        class Meta:
            database = db

    db.connect()
    # db.create_tables([Person, Pet])

    from datetime import date
    me = Person(name='max', birthday=date(1998, 7, 8), gender=True)
    modified_1 = me.save()
    my_pet = Pet(owner=me, name='a', type='b')
    modified_2 = my_pet.save()
    print(modified_1, modified_2)  # 1 1
    print(me, my_pet)

    me = Person.select().where(Person.name == 'max').get()
    my_pet = Person.select().where(Pet.owner == me).get()
    print(me, my_pet)
