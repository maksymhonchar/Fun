# src: https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989

import numpy as np
from sklearn.svm import SVC

# Points in X; classes they belong to in y.
X = np.array([ [-1, -1], [-2, -1], [1, 1], [2, 1] ])
y = np.array([1, 1, 2, 2])
# Train SVM model with the {X|y} set. Use linear kernel.
clf = SVC(kernel='linear')
clf.fit(X, y)

prediction = clf.predict([ [0, 6] ])
prediction_2 = clf.predict([ [-1, 0.5] ])
print(prediction, prediction_2)