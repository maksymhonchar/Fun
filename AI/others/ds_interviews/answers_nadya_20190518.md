# Q1
They learn a hierarchy of if/else questions, leading to a decision.

Learning a decision tree means learning the sequence of if/else questions that gets us
to the true answer most quickly

these questions are called tests. example: “Is feature i larger than value a?”

# Q2
induction + pruning

To build a tree, the algorithm searches over all possible tests and finds the one that is
most informative about the target variable - the one which shows the most information (which best separates the points).

this is a recursive process - repeating the process of looking for the best test in both regions.

# Q3
1. stopping the creation of the tree early (also called pre-pruning)
2. building the tree but then removing or collapsing nodes that contain little information (also called post-pruning or just pruning)

Possible criteria for pre-pruning include limiting the maximum depth of the tree,
limiting the maximum number of leaves, or requiring a minimum number of points
in a node to keep splitting it.

# Q4
Begin with your training dataset, which should have some feature variables and classification or regression output.
Determine the “best feature” in the dataset to split the data on; more on how we define “best feature” later
Split the data into subsets that contain the possible values for this best feature. This splitting basically defines a node on the tree i.e each node is a splitting point based on a certain feature from our data.
Recursively generate new tree nodes by using the subset of data created from step 3. We keep splitting until we reach a point where we have optimised, by some measure, maximum accuracy while minimising the number of splits / nodes.

# Q5
Because of the nature of training decision trees they can be prone to major overfitting. Setting the correct value for minimum number of instances per node can be challenging. Most of the time, we might just go with a safe bet and make that minimum quite small, resulting in there being many splits and a very large, complex tree. The key is that many of these splits will end up being redundant and unnecessary to increasing the accuracy of our model.

Tree pruning is a technique that leverages this splitting redundancy to remove i.e prune the unnecessary splits in our tree. From a high-level, pruning compresses part of the tree from strict and rigid decision boundaries into ones that are more smooth and generalise better, effectively reducing the tree complexity. The complexity of a decision tree is defined as the number of splits in the tree.

A simple yet highly effective pruning method is to go through each node in the tree and evaluate the effect of removing it on the cost function. If it doesn’t change much, then prune away!

# Q6
max_depth: The max depth of the tree where we will stop splitting the nodes. This is similar to controlling the maximum number of layers in a deep neural network. Lower will make your model faster but not as accurate; higher can give you accuracy but risks overfitting and may be slow.
min_samples_iisplit: The minimum number of samples required to split a node. We discussed this aspect of decision trees above and how setting it to a higher value would help mitigate overfitting.i
max_features: The number of features to consider when looking for the best split. Higher means potentially better results with the tradeoff of training taking longer.
min_impurity_split: Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold. This can be used to tradeoff combating overfitting (high value, small tree) vs high accuracy (low value, big tree).
presort: Whether to presort the data to speed up the finding of best splits in fitting. If we sort our data on each feature beforehand, our training algorithm will have a much easier time finding good values to split on.

# Q7
Pros
Easy to understand and interpret. At each node, we are able to see exactly what decision our model is making. In practice we’ll be able to fully understand where our accuracies and errors are coming from, what type of data the model would do well with, and how the output is influenced by the values of the features. Scikit learn’s visualisation tool is a fantastic option for visualising and understanding decision trees.
Require very little data preparation. Many ML models may require heavy data pre-processing such as normalization and may require complex regularisation schemes. Decision trees on the other hand work quite well out of the box after tweaking a few of the parameters.
The cost of using the tree for inference is logarithmic in the number of data points used to train the tree. That’s a huge plus since it means that having more data won’t necessarily make a huge dent in our inference speed.

Cons
Overfitting is quite common with decision trees simply due to the nature of their training. It’s often recommended to perform some type of dimensionality reduction such as PCA so that the tree doesn’t have to learn splits on so many features
For similar reasons as the case of overfitting, decision trees are also vulnerable to becoming biased to the classes that have a majority in the dataset. It’s always a good idea to do some kind of class balancing such as class weights, sampling, or a specialised loss function.

# Q8
For a regression tree, we can use a simple squared error as our cost function:
E=sum(y - y_hat)^2
Where Y is our ground truth and Y-hat is our predicted value
sum over all the samples in our dataset to get the total error.

for classification, use Gini Index Function
E=sum(p_k * (1-p_k))
pk are the proportion of training instances of class k in a particular prediction node. A node should ideally have an error value of zero, which means that each split outputs a single class 100% of the time.
