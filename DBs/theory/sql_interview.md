# SQL: EXISTS

SELECT column_name(s)
FROM table_name
WHERE EXISTS
(SELECT column_name FROM table_name WHERE condition);

- If a subquery returns any rows at all, EXISTS subquery is TRUE, and NOT EXISTS subquery is FALSE. For example:
        - SELECT column1 FROM t1 WHERE EXISTS (SELECT * FROM t2);

- Example: What kind of store is present in one or more cities?
SELECT DISTINCT store_type FROM stores
  WHERE EXISTS (SELECT * FROM cities_stores
                WHERE cities_stores.store_type = stores.store_type);
                
- Example 2:
select
   book_key
from
   book
where 
   exists (select book_key from sales);


# SQL: EXISTS vs IN
1. Avoid counting
--this statement needs to check the entire table
select count(*) from [table] where ...
--this statement is true as soon as one match is found
exists ( select * from [table] where ... )

This is most useful where you have if conditional statements, as exists can be a lot quicker than count.

2. The in is best used where you have a static list to pass:
select * from [table]
where [field] in (1, 2, 3)

When you have a table in an in statement it makes more sense to use a join, but mostly it shouldn't matter. The query optimiser should return the same plan either way.

3. You can also use query results with the IN clause, like this:
SELECT * 
FROM Orders 
WHERE ProductNumber IN (
    SELECT ProductNumber 
    FROM Products 
    WHERE ProductInventoryQuantity > 0)

4. Speed comparison:
EXISTS is much faster than IN, when the sub-query results is very large.
IN is faster than EXISTS, when the sub-query results is very small.


# SQL: varchar vs nvarchar
- varchar: non-Unicode, 8-bit codepage.
- nvarchar: Unicode 

With cheap disk and memory nowadays, there is really no reason to waste time with saving code pages anymore.

By using nvarchar rather than varchar, you can avoid doing encoding conversions every time you read from or write to the database

# SQL: select distinct on 
- For each group of duplicates, keep THE FIRST ROW in the returned result set
- SELECT
   DISTINCT ON (column_1) column_alias,
   column_2
FROM
   table_name
ORDER BY
   column_1,
   column_2;
- The order of rows returned from the SELECT statement is unpredictable therefore the “first” row of each group of the duplicate is also unpredictable. It is good practice to always use the ORDER BY clause with the DISTINCT ON(expression) to make the result set obvious.

# SQL: FETCH FIRST N ROWS ONLY VS LIMIT
- The following statements are equivalent:
   - SELECT * FROM foo LIMIT 10;
   - SELECT * FROM foo FETCH FIRST 10 ROWS ONLY;
- FETCH FIRST X ROWS ONLY is part of the SQL standard, while LIMIT is not.
- LIMIT is very popular, and much more terse, so it is also supported by postgres.

# Postgres: ILIKE VS LIKE
- PostgreSQL provides the ILIKE operator that acts like the LIKE operator.
- In addition, the ILIKE operator matches value case-insensitively. 
- Example:
SELECT
   first_name,
   last_name
FROM
   customer
WHERE
   first_name ILIKE 'BAR%';
The  BAR% pattern matches any string that begins with BAR, Bar, BaR, etc. If you use the LIKE operator instead, the query will not return any row.

ALSO:
- ~~ is equivalent to LIKE
- ~~* is equivalent to ILIKE
- !~~ is equivalent to NOT LIKE
- !~~* is equivalent to NOT ILIKE

# SQL: HAVING without GROUP BY
- In PostgreSQL, you can use the HAVINGclause without the GROUP BY clause.
- In this case, the HAVING clause will turn the query into a single group.
- In addition, the SELECTlist and HAVINGclause can only refer to columns from within aggregate functions.
- This kind of query returns a single row if the condition in the HAVINGclause is true or zero row if it is false.
