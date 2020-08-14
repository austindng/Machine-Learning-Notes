# based on correlation to predict data
# essentially a best fit line (y = mx + b)

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())
# prints the beginning of the data file for the students

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# all should be integers or converted to integers
# these are attributes of the student

predict = "G3"
# can be any attribute you are looking for

x = np.array(data.drop([predict], 1))
# return new data frame w/o G3 to predict another value
y = np.array(data[predict])

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

print("Coeff: \n", + linear.coef_)
# shows the coefficients for the dimensions
# bigger coeff = bigger impact on the success of student
print("Intercept: \n", + linear.intercept_)

predictions = linear.predict(x_test)
# take array of arrays do ton of tests and see what input was

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
# first num is predicted, last is actual



