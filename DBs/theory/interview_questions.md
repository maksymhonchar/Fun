# Grouping functions in SQL

- Group functions are mathematical functions to operate on sets of rows to give one result per set. The types of group functions (also called aggregate functions) are:
    - AVG
    - COUNT
    - MAX
    - MIN
    - STDDEV - standard deviation
    - VARIANCE

- The general syntax for using Group functions:
    SELECT <column, ...>, group_function(column)
    FROM <table>
    WHERE <condition>
    [GROUP BY <column>]
    [ORDER BY <column>]
    - NOTE: column on which the group function is applied must exist in the SELECT column list.
 - Example:
    SELECT dept, MAX(sal)
    FROM employee;

- To exclude some group results, use Having-clause.
    SELECT dept, MAX(sal)
    FROM employee
    GROUP BY dept
    HAVING MAX(sal) > 2000;

# Joins
-   SELECT table1.column1, table2.column2...
        FROM table1
    INNER JOIN table2
        ON table1.common_field = table2.common_field;


- INNER JOIN gets all records that are common between both tables based on the foreign key.
- LEFT JOIN gets all records from the LEFT linked table but if you have selected some columns from the RIGHT table, if there is no related records, these columns will contain NULL.
- RIGHT JOIN is like the above but gets all records in the RIGHT table.
- FULL JOIN gets all records from both tables and puts NULL in the columns where related records do not exist in the opposite table.

# UNION
- The UNION command is used to select related information from two tables, which is like a JOIN command.
- However, when using UNION command, all the selected columns need to be of the same data type. With UNION, only distinct values are selected.

- UNION ALL selects all the values.
    - It won't eliminate duplicate rows, intead it just pulls all the rows from all the tables fitting your query specifics and combines them into a table.
- UNION ALL might will give results faster - so use it if you know that all the records returned from UNION are unique.

# Joins vs UNION
- Joins and Unions can be used to combine data from one or more tables. The difference lies in how the data is combined.

- UNION puts lines from queries after each other, while JOIN makes a cartesian product and subsets it -- completely different operations.
- Union example:
mysql> SELECT 23 AS bah
    -> UNION
    -> SELECT 45 AS bah;
+-----+
| bah |
+-----+
|  23 | 
|  45 | 
+-----+
- Join example:
mysql> SELECT * FROM 
    -> (SELECT 23 AS bah) AS foo 
    -> JOIN 
    -> (SELECT 45 AS bah) AS bar
    -> ON (33=33);
+-----+-----+
| foo | bar |
+-----+-----+
|  23 |  45 | 
+-----+-----+

- UNION combines the results of two or more queries into a single result set that includes all the rows that belong to all queries in the union.
- By using JOINs, you can retrieve data from two or more tables based on logical relationships between the tables. Joins indicate how SQL should use data from one table to select the rows in another table.

- Expl 2:
    - joins combine data into new columns
    - Unions combine data into new rows