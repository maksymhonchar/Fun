import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.decomposition import NMF, PCA


S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()

# assume we have many different ways to observe the mixture - 100 measurement devices
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("Shape of measurements: {}".format(X.shape))  # (2000, 100)

# Use NMF to recover the three signals
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print(S_.shape)  # (2000, 3)

# Use PCA to recover the three signals - for comparison
pca = PCA(n_components=3)
H = pca.fit_transform(X)
print(H.shape)  # (2000, 3)

# Show the signal activity that was discovered by NMF and PCA
models = [X, S, S_, H]
names = [
    'Observations (first three measurements)',
    'True sources',
    'NMF recovered signals',
    'PCA recovered signals'
]
fig, axes = plt.subplots(
    4, figsize=(8, 4), 
    gridspec_kw={'hspace': .5}, 
    subplot_kw={ 'xticks': (), 'yticks': () }
)
for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')
plt.show()
