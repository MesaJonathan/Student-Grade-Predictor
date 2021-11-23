import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#takes in csv file and stores in pandas datafram
data = pd.read_csv("student-mat.csv", sep = ";")

#omits everything from data except for these six desired atributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#variable predict = G3 aka label
predict = "G3"

#return new dataframe withouy g3, training data, has all attributes
x = np.array(data.drop([predict], 1))
#all labes
y = np.array(data[predict])

#takes x and y, splits them up into 4 different arrays
#x and y train are dections of X and Y
#tests test for accuracy
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

#creates empty linear regression model frame
linear = linear_model.LinearRegression()

#fits linear regression model to fix data from x_train and y_train
linear.fit(x_train, y_train)
#computes accuracy of linear model relative to real values
acc = linear.score(x_test, y_test)

print(acc)

#prints coeffiecents of slope for 5d projection and intercept
print("Co: \n", linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
