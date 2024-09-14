import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
import matplotlib as mpl

def sigmoid(z):
    return 1/(1+np.exp(np.negative(z)))

def logistic_model(X, weight):
    z = np.dot(X, weight.T)
    y_pred = sigmoid(z)
    return y_pred

def cross_entropy_error(y, y_pred):
    m = len(y)
    cost = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y)*np.log(1 - y_pred))
    return cost 

def cost_function(x, y):
    cost = 0
    if y == 1:
        cost = np.negative(np.log10(logistic_model(x)))
    elif y == 0:
        cost = np.negative(np.log10(1- logistic_model(x)))
    else:
        cost = None
    return cost

X = np.array([-3, -1])
w = np.array([1, 1])
print(logistic_model(X, w))
print(sigmoid(X))