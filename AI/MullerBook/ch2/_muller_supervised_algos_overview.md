## Nearest neighbors
For small datasets, good as a baseline, easy to explain.

## Linear models
Go-to as a first algorithm to try, good for very large datasets, good for very highdimensional data.

## Naive Bayes
Only for classification. Even faster than linear models, good for very large datasets and high-dimensional data. Often less accurate than linear models.

## Decision trees
Very fast, don’t need scaling of the data, can be visualized and easily explained.

## Random forests
Nearly always perform better than a single decision tree, very robust and powerful. Don’t need scaling of data. Not good for very high-dimensional sparse data.

## Gradient boosted decision trees
Often slightly more accurate than random forests. Slower to train but faster to predict than random forests, and smaller in memory. Need more parameter tuning than random forests.

## Support vector machines
Powerful for medium-sized datasets of features with similar meaning. Require scaling of data, sensitive to parameters.

## Neural networks
Can build very complex models, particularly for large datasets. Sensitive to scaling of the data and to the choice of parameters. Large models need a long time to train.