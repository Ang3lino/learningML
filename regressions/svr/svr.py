#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# load the dataframe
df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:2] # all rows, from 1 until 2 (2 is exclusive)
y = df.iloc[:, 2:3]

# Split dataset in training, testing n dimensional arrays, if required
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# scale variables, SVR will not work if we omit this step
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# fit regression
regression = SVR(kernel="rbf")
regression.fit(X, y)

# predict the data, without forgeting to scale the independent variable
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))

# data display
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

