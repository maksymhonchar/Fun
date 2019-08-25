# src
- https://www.essentialsql.com/what-is-the-difference-between-a-join-and-subquery/

# General
- A common use for a subquery may be to calculate a summary value for use in a query.
    - For instance we can use a subquery to help us obtain all products have a greater than average product price.
    ~~~~
    SELECT  pid,
            Name,
            LastPrice,
            (SELECT AVG(ListPrice) FROM Production.Product) AS AvgListPrice
    FROM    Production.Product
    WHERE   ListPrice > (SELECT AVG(ListPrice) FROM Production.Product);
    ~~~~
- In contrast, here is the same JOIN query - combine rows from one or more tables based on a match conditioon.
    - Display product names and models
    ~~~~
    SELECT Product.Name, ProductModel.Name
    FROM Production.product
    INNER JOIN Production.ProductModel
    ON Product.ProductModelID = ProductModel.ProductModelID
    ~~~~

# Differences
- Return values:
    - Subqueries can be used to return either a scalar (single) value or a row set;
    - Joins are used to return rows.