# src
- http://www.postgresqltutorial.com

# Postgres: GROUPING SETS
- A grouping set is a set of columns by which you group.
- Typically, a single aggregate query defines a single grouping set.

- GROUPING SETS is the subclause of the GROUP BY clause
- The GROUPING SETS allows you to define multiple grouping sets in the same query.

- The general syntax of the GROUPING SETS is as follows:
SELECT
    c1,
    c2,
    aggregate_function(c3)
FROM
    table_name
GROUP BY
    GROUPING SETS (
        (c1, c2),
        (c1),
        (c2),
        ()
);

# GROUPING
- The GROUPING function accepts a name of a column and returns bit 0 if the column is the member of the current grouping set and 1 otherwise. 
- Example:
SELECT
   GROUPING(brand) grouping_brand,
   GROUPING(segment) grouping_segment,
   brand,
   segment,
   SUM (quantity)
FROM
   sales
GROUP BY
   GROUPING SETS (
      (brand, segment),
      (brand),
      (segment),
      ()
   )
ORDER BY
   brand,
   segment;

# Postgres: CUBE
- PostgreSQL CUBE is a subclause of the GROUP BY clause.
- The CUBE allows you to generate multiple grouping sets.
    - A grouping set is a set of columns to which you want to group: GROUPING SETS ( (..), .. )

- The query generates all possible grouping sets based on the dimension columns specified in CUBE.

- The CUBE subclause is a short way to define multiple grouping sets so the following are equivalent:
CUBE(c1,c2,c3) 
==
GROUPING SETS (
    (c1,c2,c3), 
    (c1,c2),
    (c1,c3),
    (c2,c3),
    (c1),
    (c2),
    (c3), 
    ()
) 
- In general, if the number of columns specified in the CUBE is n, then you will have 2n combinations.

- Example:
SELECT
    c1,
    c2,
    c3,
    aggregate (c4)
FROM
    table_name
GROUP BY
    CUBE (c1, c2, c3);  -- dimension columns we want to analyze: c1 c2 c3

# Postgres: ROLLUP
- The PostgreSQL ROLLUP is a subclause of the GROUP BY clause that offers a shorthand for defining multiple grouping sets

- Different from the CUBE subclause, ROLLUP does not generate all possible grouping sets based on the specified columns -> It just makes a subset of those

- The ROLLUP assumes a hierarchy among the input columns and generates all grouping sets that make sense considering the hierarchy. This is the reason why ROLLUP is often used to generate the subtotals and the grand total for reports.

- A common use of  ROLLUP is to calculate the aggregations of data by year, month, and date, considering the hierarchy year > month > date

- Comparison of CUBE vs ROLLUP:
    - CUBE (c1,c2,c3)
    (c1, c2, c3)
    (c1, c2)
    (c2, c3)
    (c1,c3)
    (c1)
    (c2)
    (c3)
    ()
    - ROLLUP(c1,c2,c3)  -- assume the hierarchy c1>c2>c3
    (c1, c2, c3)
    (c1, c2)
    (c1)
    ()

- Syntaxis example:
SELECT
    c1,
    c2,
    c3,
    aggregate(c4)
FROM
    table_name
GROUP BY
    ROLLUP (c1, c2, c3);

- Example: find the number of rental per day, month and year by using the ROLLUP
SELECT
    EXTRACT (YEAR FROM rental_date) y,
    EXTRACT (MONTH FROM rental_date) M,
    EXTRACT (DAY FROM rental_date) d,
    COUNT (rental_id)
FROM
    rental
GROUP BY
    ROLLUP (
        EXTRACT (YEAR FROM rental_date),
        EXTRACT (MONTH FROM rental_date),
        EXTRACT (DAY FROM rental_date)
    );
