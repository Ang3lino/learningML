#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 18:08:23 2019

@author: angelos
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def plot_2d_plane(x, y, y_pred, title, xlabel='x', ylabel='y'):
    plt.scatter(x, y_pred, color='red')
    plt.plot(x, y, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# load the dataframe
df = pd.read_csv("Position_Salaries.csv")
x = df.iloc[:, 1:2] # all rows, from 1 until 2 (2 is exclusive)
y = df.iloc[:, 2:3]

# fit polynomial regression with the dataset
# here, x_poly will be set into a polynomial matrix from the vector column x
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x) 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
y_pred = lin_reg_2.predict(x_poly)

# contrast the two regression models
plot_2d_plane(x, y, y_pred, title='Polynomial regression')
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)
plot_2d_plane(x, y, y_pred, title='Linear regression')
