
#%%
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

import math

from sklearn import datasets
from sklearn.model_selection import train_test_split


#%%
# load dataframe, split it into train, test
df = datasets.load_iris()
X = df.data[:, :2]
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


#%%
def euclidean_distance(x, y):
    return np.sqrt(np.sum(((x - y) ** 2)))

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y)) 

#%%
def __knn_predict(x, y, p, k):
    n = len(x)
    class_dist = [(y[i], euclidean_distance(x[i], p)) for i in range(n)]
    class_dist_sorted_by_distance = sorted(class_dist, key=lambda kv: kv[1])[:k]  
    get_classification = lambda x: x[0]
    classes = list(map(get_classification, class_dist_sorted_by_distance))
    unique_classifications = set(map(get_classification, class_dist_sorted_by_distance))
    frequencies = [(c, classes.count(c)) for c in unique_classifications]
    return max(frequencies, key=lambda cd: cd[1])[0]

def knn_predict(X_train, y_train, X_test, k=5):
    return [__knn_predict(X_train, y_train, x, k) for x in X_test]
    
def accurracy(xtrain, y_train, xtest, y_test, k=5):
    count = 0
    n = len(y_test)
    y_pred = knn_predict(xtrain, y_train, xtest, k)
    for i in range(n):
        if y_pred[i] == y_test[i]:
            count += 1
    return (count / n) * 100


print(accurracy(X_train, y_train, X_test, y_test, 5))

#%%
assert(euclidean_distance(np.array([0, 0]), np.array([3, 4])) == 5.0)

#%%