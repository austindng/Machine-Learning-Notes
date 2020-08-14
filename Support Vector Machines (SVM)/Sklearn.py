# classify data w/o linear correspondence
# works well for hi-dimensional data
# hard margins:
# divides data into multiple classes using a hyper-plane
# picks 2 points as support vectors with identical dist.
# usually with the greatest possible margin
# kernels bring in another dimension (ie: 2D to 3D)
# soft margins: allows outliers so hyperplane would perform better

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# sees all features
print(cancer.target_names)
# sees all targets

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(y_train)
# shows data as 0 or 1
# 0 is malignant
# 1 is benign

classes = ["malignant", "benign"]

clf = svm.SVC(kernel="linear", C=2)
# kernel is the type of eq.
# C is how big the soft margin is going to be
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
