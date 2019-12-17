# Foreign Key Referential Integrity
- In PostgreSQL, you define a foreign key through a foreign key constraint.
- A foreign key constraint indicates that values in a column or a group of columns in the child table match with the values in a column or a group of columns of the parent table.
- We say that a foreign key constraint maintains referential integrity between child and parent tables.

- Referential integrity is a relational database concept that states implied relationships among data should be enforced.
- Referential integrity ensures that the relationship between rows in two tables will remain synchronized during all updates and deletes.

# Views
- A view is a virtual table that is used to simplify complex queries and to apply security for a set of records.
- PostgreSQL also provides you with updatable views.

- A view is named query that provides another way to present data in the database tables.
- A view is defined based on one or more tables, which are known as base tables.
- When you create a view, you basically create a query and assign it a name, therefore a view is useful for wrapping a commonly used complex query.

- Note that a normal view does not store any data except the materialized view.
- In PostgreSQL, you can create a special view called a materialized view that stores data physically and refreshes the data periodically from the base tables.
- The materialized views have many advantages in many scenarios such as faster access to data from a remote server, data caching, etc.

- CREATE VIEW view_name AS query;
- CREATE VIEW usa_cities AS SELECT
   city,
   country_id
FROM
   city
WHERE
   country_id = 103;
- CREATE MATERIALIZED VIEW view_name
    AS
    query
    WITH [NO] DATA;
- REFRESH MATERIALIZED VIEW [CONCURRENTLY] view_name;

# Schemas
- A schema is a logical container of tables and other objects inside a database.
- Each PostgreSQL database may have multiple schemas.
- It is important to note that schema is a part of the ANSI-SQL standard.

# Tablespaces
- A tablespace is where PostgreSQL stores the data.
- PostgreSQL tablespace enables you to move your data to different physical location across drivers easily by using simple commands.
- By default, PostgreSQL provides two tablespaces: pg_default for storing user’s data and pg_global for storing system data.

# Functions
- A function is a block reusable SQL code that returns a scalar value of a list of records.
- In PostgreSQL, functions can also return composite objects.

# Extensions
- PostgreSQL introduced extension concept since version 9.1 to wrap other objects including types, casts, indexes, functions, etc., into a single unit.
- The purpose of extensions is to make it easier to maintain.
