
# -*- coding: utf-8 -*-
"""
@author: Angel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from matplotlib.colors import ListedColormap

# import csv file
df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2, 3]].values  # pick age, salary
y = df.iloc[:, -1].values  # did he bought it?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

regressor = LogisticRegression(random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


def heatmap(X_test, y_test, classifier):
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Clasificador (Conjunto de Entrenamiento)')
    plt.xlabel('Edad')
    plt.ylabel('Sueldo Estimado')
    plt.legend()
    plt.show()


heatmap(X_test, y_test, regressor)