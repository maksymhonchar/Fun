# Q1
- To build a machine learning model initially you had 100 data points and 5 features. To reduce bias, you doubled the features to include 5 more variables and collected 100 more data points. Explain if this is a right approach? (Hint: There is a curse on machine learning. Have you heard about it?)

## general q1
- In general, adding more data == good and helps to decrease overfitting
    - Adding more examples, adds diversity. It decreases the generalization error because your model becomes more general by virtue of being trained on more examples.
- BUT: if additional training data is noisy or doesn't match whatever you are trying to predict = very bad

- Adding more features: may increase overfitting because more features may be either irrelevant or redundant and there's more opportunity to complicate the model in order to fit the examples at hand.

- High bias – a simpler model that doesn’t tend to overfit, but may underfit training data, failing to capture important regularities.

- bias-variance tradeoff
see picture q1_img.png

## answer q1
- Curse of dimensionality: as the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially
- Example: 1 dimension 10 points; 2 dimensions: 100 points; d dimensions: 10^d points.
- Before points: 100^5=100000; after: 200^10=102400000000000000000000

- Example: get a population, predict some property:
    - instances represented as {urefu, height} pairs
    - what is the dimensionality of the data?
    - BUT!!! "height"="urefu" in Swahili
    - so is it only 1 dimension?!
    - What this example shows - know what your dimensions represent
- Example 2: data points over time from different geographic ares over time:
    - x1 - # of skidding accidents
    - x2 - # of burst water pipes
    - x3 - snow-plow expenditures
    - x4 - # of school closures
    - xn - # patients with heat stroke
    - WHAT TO NOTE: there is one factor that could possibly explain all of these things
        - answer - the temperature!

# Q2
- Let’s say you have a extremely small memory/storage. What kind of algorithm would you prefer — logistic regression or k-nearest neighbor? Why? (Hint: Space complexity)

## general q2
- space complexity - metric for how much storage space the algorithm needs in relation to its inputs.

## answer q2
- kNN: O(N) - we have to store all N samples
- logreg: O(N) - That is we need to store weight vector for each dimension.

- size(weight vector for each dimension) < size(all samples) so the answer is LogReg

# Q3
- Imagine your data set is known to be linearly separable and you have to guarantee the convergence and maximum number of iterations/steps of your algorithm (due to computational resource reason). Would you choose gradient descent in this case? What can you choose? (Hint: Which simple algorithm provides guarantee of finding solution?)

## general q3
- Gradient Descent is an algorithm which is designed to find the optimal points, but these optimal points are not necessarily global. 
- If it happens that it diverges from a local location it may converge to another optimal point but its probability is not too much.
- The reason is that the step size might be too large that prompts it recede one optimal point and the probability that it oscillates is much more than convergence.

- If tuned properly, gradient descent can take you to the vicinity of a global minimizer for convex optimization problems in a finite number of steps.

- Safeguarded methods - are usually based on trust regions or line search, and are meant to ensure convergence to something

- Linear programming (LP, also called linear optimization) is a method to achieve the best outcome (such as maximum profit or lowest cost) in a mathematical model whose requirements are represented by linear relationships

- OTHER OPTIMIZATION ALGORITHMS worth to consider:
    - Newton optimization method
    - Quasi-Newton method
        - BFGS
        - L-BFGS
    - RMSprop
    - Adam: adaptive moment estimation

## answer q3
- Would you choose gradient descent in this case - NO
- wiki: In practice, the simplex algorithm is quite efficient and can be guaranteed to find the global optimum if certain precautions against cycling are taken
- However, the simplex algorithm has poor worst-case behavior: Klee and Minty constructed a family of linear programming problems for which the simplex method takes a number of steps exponential in the problem size

# Q4
- If you could take advantage of multiple CPU cores, would you prefer a boosted-tree algorithm over a random forest? Why? (Hint: if you have 10 hands to do a task, you take advantage of it)

## general q4
- Any incoming datum is classified by all trees, then the trees vote for which classification the datum recieves. The most common classification is given to the datum as its predicted label. The more trees, the better.

- Any algorithm that relies on matrix multiplication (or, in general, any parallelizable operation) can be accelerated by performing those operations on GPU.

- The other thing to consider is whether or not the boost is worth it. For example, logistic regression could be sped up on gpus but they're probably fast enough on cpus.

- both random forest and xgboost are examples of ensemble:
    - random forest - bagging
    - xgboost - boosting

## answer q4
- so the answer is random forest
    - boosted trees -> GPU, random forest -> CPU

- Example: xgboost is an example of a ml library that exploits parallelism - anything with a lot of flops or memory bandwidth being the bottleneck can benefit from gpus.

- The overhead becomes more significant than the gains in performance.

- In random forest, decision trees are created paralelly
- There, we give a sample of training dataset for each decision tree
- We train each model (each tree), find the accuracy for each model
- Get the average of the accuracies found

- In xgboost, we SEQUENTIAL decision tree - it is not created parallely, but sequentially
- because it is sequential, we have to wait until every single one will be built == finished
- unlike random forest - we build every tree independently

# Q5
- Which is more important to you – model accuracy, or model performance?

## general q5
- Accuracy is the number of correct predictions made by the model by the total number of records

- Accuracy is actually one of the metrics of performance measures of a model

- ROC curve: shows the trade-off between the TP rate and FP rate
    - TP rate: 1->1 (positive come out to be positive)
        - someone who had hancer predicted to have cancer
    - FP rate: 1->0 (negative came out to be positive)
        - someone NOT having cancer, predicted to have cancer
- The area under the ROC curve is a measure of the accuracy of the model

- Recall or sensitivity gives us information about a model’s performance on false negatives
- precision gives us information of the model’s performance of false positives

## answer q5
- if performance is time:
    - for real-time applications - model performance is much more important
        - example: object-detector on production line
        - face detection example - we want it do be really fast
    - for - model accuracy is really important
        - ie medical application to detect tumors

- if performance is precision vs recall:
    - Based on what is predicted, precision or recall might be more critical for a model.