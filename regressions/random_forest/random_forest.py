
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# load csv data
df = pd.read_csv(os.path.join('random_forest', 'Position_Salaries.csv'))
X = df.iloc[:, 1:2]  # all rows, from 1 until 2 (2 is exclusive). type(X) pandas.core.frame.DataFrame
y = df.iloc[:, 2:3]

"""
# Split data in training, testing n dimensional arrays, if required
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)
"""

rfg = RandomForestRegressor(n_estimators=10, random_state=0)
rfg.fit(X, y)
X = X.values  # type -> n dimensional array
y = y.values


def plot_2d(x, y):
    X_grid = np.arange(min(x), max(x), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1)
    plt.scatter(x, y, color="red")
    plt.plot(X_grid, rfg.predict(X_grid), color="blue")
    plt.title("random forest plot")
    plt.xlabel("Employee's level")
    plt.ylabel("Salary ($)")
    plt.show()


plot_2d(X, y)
# Be careful, if we fit with a value x which is not in [min(X), max(X)]
# the model may not predict appropriately
arr = np.array([k for k in range(11, 21)]).reshape(-1, 1)
arr = np.concatenate((X, arr))
res = rfg.predict(arr)
plot_2d(arr, res)

