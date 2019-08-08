# note: for NNs

1. Define the task and dataset
        - What kind of data are considered as inputs?
        - What should we predict?
        - What kind of task is it:
                - multiclass classification, scalar regression, vector regression,
                clusterization, generation, reinforcement learning ...
        - What are the hypothesis?
                - remember: output data could be predicted using input data
                - available data is informative enough to learn relations between input
                and output data.
                - Note: not all problems are solvable! X might contain not enough
                information to predict y.
2. Select measure of success
        - What exactly is success? Proximity, precision and recall, customer retention.
        - What is loss/cost function?
                - for symmetric classification (when each class is equally likely to
                happen) - ROC AUC 
                - non-symmetric classification: precision and recall
                - ranking tasks, multiclass classification: average, expected value
                - sometimes it will be a custom metric (i.e. kaggle metrics)
3. Select way to assess quality of the model
        - select validation set from the general set (big size of dataset)
        - k-fold crossvalidation (small size of dataset)
        - iterated k-fold validation with shuffling (small size of dataset)
4. Data preprocessing
        - data should be put into tensors
        - scale data (i.e. [-1,1], [0,1])
        - if data are in different regions - normalize them.
        - for small amount of features - perform feature selection.
5. Baseline model development
        - goal: have a high statistical power == model outputs more accurate result than
        the basic one
        - Make 3 choices for creating baseline model:
                - activation function for last level - sigmoid or without activation
                function (regression)
                - loss function - binary cross-entropy, MSE (regression)
                - optimization configuration - which optimizator to use (rmsprop with
                default step)
        - Choices:
                - binary classification: sigmoid + binary cross-entropy
                - multiclass/single class classification - softmax + categorical
                cross-entropy
                - multiclass classification: sigmoid + binary cross-entropy
                - regression for random values: None + MSE
                - regression for [0,1] values: sigmoid + MSE/binary cross-entropy
6. Overfitted model development
        - Are there enough layers and parameters to correctly model the problem?
        - Ideal model - the one, which stands on the brink of over and underfitting
                - plot maybe?
        - First we should construct overfitted model:
                - Add more layers
                - Add more features for the layers
                - Train model for a big amount of epochs
        - Always control how loss function moves on each step of training
7. Regularization model and hyperparameters tuning
        - Change your model multiple time, retrain in, assess quality on validation
        dataset (WITHOUT using training data!)
        - Repeat the cycle until model accuracy reaches desired level of accuracy.
        - To try:
                - optimal brain surgery (Оптимальное прореживание нейронных сетей)
                - try different architectures: add and remove layers
                - add L1 or L2 regularization
                - Try different hyperparameters (# of neurons in layers, optimizator
                learning step) to find optimal settings
                - Add new features / remove existing ones which are non informative
