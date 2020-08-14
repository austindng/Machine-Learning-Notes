# unsupervised learning algorithm
# divides training data into k unique cluster to classify info
# places a number of centroids(k) randomly on plot
# finds halfway pt and slices data in half
# categorizes one side over another with the mean of the distance
# does this until no pts are changed from categories

import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)
# all the features from -1 to 1
# they will all have large values
# scaling down will save time for computation

y = digits.target
k = 10
# or k = len(np.unique(y))

samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

