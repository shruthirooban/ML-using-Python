#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:25:09 2019

@author: shruthisreenivasamurthy


IMPORTING THE LIBRARIES

"""
import numpy as np
import matplotlib as mpt
import pandas as pd
import datetime as dt

""""IMPORTING DATASET""""

dataset = pd.read_excel("Departures.xlsx")

X = dataset.iloc[:, :-1].values
print (X)

Y = dataset.iloc[:, 9].values
print (Y)

""""MISSING VALUES""""

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis=0)
imputer = imputer.fit(X[:, 0:9])
X[:, 0:9] = imputer.transform(X[:, 0:9])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

""""labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()""""

print (X)

""""SPLITTING THE DATASET INTO TEST DATA AND TRAIN DATA""""

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


""""LINEAR REGRESSION""""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
y_pred = regressor.predict(X_train)

import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], y_pred, color = 'purple')
plt.scatter(X_train[:, 0], Y_train, color = 'red')
plt.title("Predicting Delay in Arrivals (training set)")
plt.xlabel("Hour of the Day")
plt.ylabel("Delays")
plt.show()

plt.scatter(X_test[:, 2], Y_pred, color = 'purple')
plt.scatter(X_test[:, 2], Y_test, color = 'red')
plt.title("Predicting Delay in Arrivals (training set)")
plt.xlabel("Month of the Year")
plt.ylabel("Delays")
plt.show()

""""MULTINOMIAL LINEAR REGRESSION""""

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((82673, 1)).astype(int), values = X, axis = 1)

print (X)

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5, 6, 7, 8, 9]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


""""PLOYNOMIAL LINEAR REGRESSION""""

X = dataset.iloc[:, 0:9].values
print (X)

Y = dataset.iloc[:, 9].values
print (Y)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis=0)
imputer = imputer.fit(X[:, 0:9])
X[:, 0:9] = imputer.transform(X[:, 0:9])

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

plt.scatter(X[:, 0], Y, color = 'red')
plt.scatter(X[:, 0], lin_reg.predict(X), color = 'purple')
plt.title("Linear Regression - Hour VS Delay")
plt.xlabel("hour of the day")
plt.ylabel("delay")
plt.show()


X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X[:, 0], Y, color = 'red')
plt.scatter(X[:, 0], lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'purple')
plt.title("Polynomial of Hour VS Delay")
plt.xlabel("hour of the day")
plt.ylabel("delay")
plt.show()

lin_reg.predict(X)
lin_reg_2.predict(poly_reg.fit_transform(X))

""""Decision Tree Regression""""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X[:, 0:3], Y)


Y_pred = regressor.predict(X[:, 0:3])

plt.scatter(X[:, 0], Y, color = 'red')
plt.scatter(X[:, 0], regressor.predict(X[:, 0:3]), color = 'purple')
plt.title("Decision Tree Regression of Hour VS Delay")
plt.xlabel("hour of the day")
plt.ylabel("delay")
plt.show()


X_grid = np.arange(min(X[:, 0]), max(X[:, 3]), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X[:, 0], Y, color = 'red')
plt.scatter(X[:, 0], regressor.predict(X[:, 0:3]), color = 'purple')
plt.title("Decision Tree - Hour VS Delay")
plt.xlabel("hour of the day")
plt.ylabel("delay")
plt.show()


""""Random Forest Regression""""

import numpy as np
import matplotlib as mpt
import pandas as pd
import datetime as dt


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X[:, 0:3], Y)

Y_pred = regressor.predict(X[:, 0:3])

plt.scatter(X[:, 0], Y, color = 'red')
plt.scatter(X[:, 0], regressor.predict(X[:, 0:3]), color = 'purple')
plt.title("Random Forest Regression of Hour VS Delay")
plt.xlabel("hour of the day")
plt.ylabel("delay")
plt.show()


""""RANDOM FOREST CLASSIFICATION""""

#Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting Random Forest Classification to the Training Set#
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train[:, 0:2], Y_train)

#Predicting the Test Set Results
Y_pred = classifier.predict(X_test[:, 0:2])

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Visualising Training Set Results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01), 
                         np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step = 0.01))

import matplotlib.pyplot as plt

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())                             
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)): plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], 
                      c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Random forest Classification (Traning Set)")
plt.legend()
plt.show()


""""Naive Bayes Classification""""

#Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting Naive Bayes Classification to the Training Set#
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

#Predicting the Test Set Results
Y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Visualising Training Set Results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
                         np.arange(start = X_set[:, 3].min() - 1, stop = X_set[:, 3].max() + 1, step = 0.01))

import matplotlib.pyplot as plt

for i, j in enumerate(np.unique(Y_set)): plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], 
                      c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Naive Bayes Classification (Traning Set)")
plt.legend()
plt.show()





















