# if >90% this will save it from fluctuating for better accuracy
# better for large data sets that take longer than 1 sec to train

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())
# prints the head of data of students

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# all should be integers or converted to integers
# these are attributes of the student

predict = "G3"
# label = what you are looking for

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# return new data frame w/o G3 to predict another value

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

'''''''''''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
    # splitting att into 4 diff tests
    # memorize patterns through testing

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    # fit this data to find best fit line
    # test test data on it

    acc = linear.score(x_test, y_test)
    print(acc)
    # accuracy

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
        # saves the model(linear) for us in our directory
        # saves the best accuracy
        '''''''''''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coeff: \n", + linear.coef_)
# shows the coefficients for the dimensions
# bigger coeff = bigger impact on the success of student
print("Intercept: \n", + linear.intercept_)

predictions = linear.predict(x_test)
# take array of arrays do ton of tests and see what input was

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
# first num is predicted, last is actual

p = 'studytime'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
# plots the points on a graph so it will be easier to view correlation

