
# -*- coding: utf-8 -*-
"""

@author: Angel
"""

# preprocessing template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# import csv file
df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""