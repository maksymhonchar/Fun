# Quick summary
- k-means and agglomerative clustering allow you to specify the number of desired clusters
- DBSCAN lets you define proximity using the eps parameter, which indirectly influences cluster size.

- Strengths:
    - k-means
        - Allows for a characterization of the clusters using the cluster means.
        - It can also be viewed as a decomposition method, where each data point is represented by its cluster center.
    - DBSCAN
        - allows for the detection of “noise points” that are not assigned any cluster
        - can help automatically determine the number of clusters.
        - allow for complex cluster shapes (muller: two_moons examplep).
        - sometimes produces clusters of very differing size, which can be a strength or a weakness.
    - Agglomerative clustering
        - can provide a whole hierarchy of possible partitions of the data, which can be easily inspected via dendrograms.

# PCA - Principal Component Analysis
- Method that rotates the dataset in a way such that the rotated features are statistically uncorrelated.

- rotation is often followed by selecting only a subset of the new features, according to how important they are for explaining the data.

- The algorithm proceeds by first finding the direction of maximum variance
    - this is the vector that contains most of the information 
- Then, it finds the direction that contains the most information while being orthogonal (at a right angle) to the first direction

- Direction found == principal components
    - they are the main directions of variance in the data.

- In general, there as many principal components as original features.

- This transformation is sometimes used to remove noise effects from the data or visualize what part of the information is retained using the principal components.

- It is important to note that PCA is an unsupervised method, and does not use any class information when finding the rotation. It simply looks at the correlations in the data.

- A downside of PCA is that the two axes in the plot are often not very easy to interpret. The principal components correspond to directions in the original data, so they are combinations of the original features.
    - However, these combinations are usually very complex.

- PCA can be useful in many situations, but especially in cases with excessive multicollinearity or explanation of predictors is not a priority.

## The PCA recipe:
1. Center the data
2. Normalize the data
3. Calculate the eigendecomposition
4. Project the data

## PCA - feature extraction
- The idea behind feature extraction is that it is possible to find a representation of your data that is better suited to analysis than the raw representation you were given.

- Use "whitening" option of PCA, which rescales the principal components to have the same scale.
- This is the same as using sklearn.StandardScaler after the transformation.

# NMF - non-negative matrix factorization
- NMF (from sklearn) - find two non-negative matrices (W, H) whose product approximates the non-negative matrix X.

- used for feature extraction

- Works similarly to PCA and can also be used for dimensionality reduction.

- In NMF, we want the components and the coefficients to be NONNEGATIVE
    - == both the components and the coeffs to be >= 0.
    - In contrast to when using PCA, we need to **ensure** that our data is positive for NMF to be able to operate on the data.

- NMF can only be applied to data where each feature is non-negative

- NMF leads to more interpretable components than PCA, as negative components and coefficients can lead to hard-to-interpret cancellation effects.

- NMF uses a random initialization, which might lead to different results depending on the random seed.

- the components produced by NMF have no natural ordering.

## PCA & NMF
What PCA and NMF do: decompose each data point into a weighted sum of a fixed set of components.

# Manifold learning t-SNE
- Purposes of manifold learning algorithms:
    - visualization using 2-dimensional scatter plots
        - use it in exploratory analysis!

- t-SNE computes a new representation of the training data, but doesn't allow transformations of new data (== new features)
- This means, these algorithms CANNOT be applied to a test set: rather, they can only transform the data they were trained for.

- Idea behind t-SNE: find a 2-dimensional representation of the data that preserves the distances between points as best as possible.
- t-SNE puts more emphasis on points that are close by, rather than preserving distances bewteen far-apart points.
    - == it tries to preserve the information indicating which points are neighbors to each other.

- t-SNE parameters:
    - perplexity
    - early_exaggeration

# Clustering
- Clstering is the task of partitioning the dataset into groups, called clusters.
    - goal - split the data in such a way that points withing a single cluster are very similar and points in different clusters are different.
- Clustering algorithms assign (or predict) a number to each data point, indicating which cluster a particular point belongs to.

## k-means clustering
- It tries to find cluster centers (centroids) that are representative of certain regions of the data
- The algorithm has 2 steps:
    1. Assign each data point to the closest cluster center
    2. Set each cluster center as the mean of the data points that are assigned to it.
- The algorithm is finished when the assignment of instances to clusters no longer changes.

- k-means scales easily to large datasets
    - sklearn: MiniBatchKMeans

- Drawbacks:
    - relies on a random initialization

## Agglomerative Clustering
- Agglomerative Clustering - collection of clustering algorithms that all build upon th esame principles:
    - the algorithm starts by declaring each point its own cluster
    - merge the 2 most similar clusters until some stopping criterion is satisfied.
        - stopping criterion in sklearn: no of clusters

- Linkage criteria: specify how exactly the "most similar cluster" is measured:
    - ward - default; pick the 2 clusters to merge such that the variance within all clusters increases the least == equally sized clusters
    - average - merge the 2 clusters that have the smallest average distance between al their points.
    - complete - maximum linkage - merge the 2 clusters that have the smallest maximum distance between their points.

- ward - works on most datasets
- average, complete - if the clusters have very dissimilar numbers of members (if one is much bigger that all the others)

- Because of the way the algorithm works, agglomerative clustering cannot make predictions for new data points.

## DBSCAN density-based spatial clustering of applications with noise.
- Main benefits:
    - it doesn't require the user to set the number of clusters
    - it can identify points that are not part of any cluster

- The idea behind DBSCAN is that clusters form dense regions of data, separated by regions that are relatively empty.

- Parameters:
    - min_sampless
    - eps

# Comparing and Evaluating Clustering algorithms
- Measure [0;1]:
    1. Adjusted rand index (ARI)
    2. Normalized mutual information (NMI)

- A common mistake when evaluating clustering in this way is to use accuracy_score instead of adjusted_rand_score , normalized_mutual_info_score , or some other clustering metric.

## Evaluating clustering without ground truth
- When applying clustering algorithms, there is usually no ground truth to which to compare the results.
    - Therefore, using metrics like ARI and NMI usually only helps in developing algorithms, not in assessing success in an application.

- Solution - **Silhouette Coefficient**
    - often doesn't work well in practice
- computes the compactness of a cluster, where higher is better, with a perfect score of 1.
- While compact clusters are good, compactness doesn’t allow for complex shapes.

- A slightly better strategy for evaluating clusters is using **robustness-based clustering metrics**.
- These run an algorithm after adding some noise to the data, or using different parameter settings, and compare the outcomes.
- The idea is that if many algorithm parameters and many perturbations of the data return the same result, it is likely to be trustworthy.

