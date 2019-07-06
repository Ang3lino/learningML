
#%%
import pandas as pd 

from sklearn import datasets
from sklearn.model_selection import train_test_split

import math, csv, functools
import statistics as stats

#%%
def list_split(src, n):
    first, rest = [], src.copy()
    for _ in range(n):
        first.append(rest.pop(0))  # push the first element from rest into first list
    return first, rest

def preprocess(filename):
    with open(filename) as fp:
        lines = fp.read()
    lines = lines.split('\n')
    names, rest = list_split(lines, 9) 

    # w+ creates a new file for writting, newline'll be set after writting a line
    with open('pid.csv', 'w+', newline='') as fp:
        writer = csv.writer(fp) 
        rows = [[float(cell) for cell in row.split(',')] for row in rest]
        writer.writerow(names)        
        writer.writerows(rows)


preprocess('pima-indians-diabetes.csv')

#%%
df = pd.read_csv('pid.csv')
X = df.iloc[:, :8].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
df.head()  # vscode displays much better in this way than passing it to print function


#%%
def split_by_class(X, y):
    result = {}  # set 
    for xi, yi in zip(X, y):
        if yi not in result:
            result[yi] = []
        result[yi].append(xi)
    return result


splitted = split_by_class(X_train, y_train)


#%%
def summarize_by_class(X, Y):
    # NOP
    splitted = split_by_class(X, Y)
    result = {}
    for clazz, row in splitted.items():
        result[clazz] = [(stats.mean(col), stats.stdev(col)) for col in zip(*row)]
    return result


summaries = summarize_by_class(X_train, y_train)
assert(len(summaries[0]) == len(X_train[0]))
summaries


#%%
def gaussian_pdf(x, mu, stdev):
    exponent = math.exp(-((x - mu) ** 2) / (2 * stdev ** 2))
    return exponent / math.sqrt(2 * math.pi * stdev ** 2)

def compute_class_probabilities(summaries, X):
    probabilities = {}
    for classification, mean_stdevs in summaries.items():
        probabilities[classification] = functools.reduce(
                lambda acc, curr: acc * curr, 
                (gaussian_pdf(x, mu, stdev) for x, (mu, stdev) in zip(X, mean_stdevs)),
                1   )
    return probabilities

def predict_one(means_stdevs, x):
    probabilities = compute_class_probabilities(summaries, x)
    return max(probabilities.items(), key=lambda arg: arg[1])[0]    

def predict(summaries, X_test):
    return [predict_one(summaries, x) for x in X_test]

def accurracy(summaries, X_test, y_test):
    y_pred = predict(summaries, X_test)
    matched = filter(lambda t: t[0] == t[1], zip(y_pred, y_test))
    n_predicted = sum(1 for _ in matched)
    return (n_predicted / len(y_test)) * 100

    
print(accurracy(summaries, X_test, y_test))  # 76.6234 accurate
            

#%%
# testing, product by using reduce and unpacking
def test():
    acc = 1
    a = [1, 4, 7]
    b = [(2, 3), (5, 6), (8, 9)]
    for i in range(len(a)):
        acc *= gaussian_pdf(a[i], b[i][0], b[i][1])

    accf = functools.reduce(
            lambda acc, curr: acc * curr, 
            (gaussian_pdf(x, mu, stdev) for x, (mu, stdev) in zip(a, b)),
            1)

    assert(acc == accf)

    # find matches given two iterables x, y
    # set(x) & set(y)