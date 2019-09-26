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
