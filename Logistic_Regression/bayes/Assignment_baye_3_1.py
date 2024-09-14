import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
import matplotlib as mpl

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def density_function(mean, X, std):
    # return (1/(np.sqrt(2*np.pi))*std)*np.exp(np.negative((X - mean)**2/(2*std**2)))
    return np.exp(np.negative((X - mean)**2/(2*std**2)))/((np.sqrt(2*np.pi))*std)

def probability_pr(X, mean, D, prior, covariance):
    y = []
    for i in range(len(X)):
        _y = ((-(0.5)*(X[i]-mean).T)*np.linalg.inv(covariance)*(X[i]-mean) - (np.abs(D)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance)) + np.log(prior))* 0.025
        y.append(np.linalg.det(_y))
        # y.append(_y)

    return y
    
    # return np.exp(np.negative(0.5)*(X-mean).T*np.linalg.inv(sum(X-mean)))/(((2*np.pi)**(D/2))*np.abs(covariance)**0.5)

def likelihood():
    pass
def evidence():
    pass
def posterior():
    pass
# def quadratic_discriminant(X, mean, attribute, sigma_class):
#     # return np.negative(0.5)*((X-mean).T)*np.linalg.inv(sigma_class)-(np.abs(attribute)/2)*np.log(2*np.pi) - 0.5*np.log(abs(sigma_class)) + np.log(quadratic_g(attribute, np.mean(attribute), len(attribute), sigma_class))-np.log(quadratic_g(X, np.mean(X), len(X), sigma_class))
#     A = np.negative(0.5)*((X-mean).T)*np.linalg.inv(sigma_class)
#     B = (np.abs(attribute)/2)*np.log(2*np.pi)
#     C = 0.5*np.log(abs(sigma_class))
#     D = np.log(probability_pr(attribute, np.mean(attribute), len(attribute), sigma_class))
#     E = np.log(probability_pr(X, np.mean(X), len(X), sigma_class))

    # return A - B - C  + D - E
mean_class1 = np.array([2, 0])
mean_class2 = np.array([1, 0])
X_range = np.linspace(-4, 4, 100)
covariance_class1 = [[0.5, 0],[0, 0.25]]
covariance_class2 = [[0.25, 0],[0, 0.5]]
prior1 = 0.5
prior2 = 0.5
attribute = 100
X = np.stack((X_range, X_range), axis=1)
Y = probability_pr(X, mean_class1, attribute, prior1, covariance_class1)
x, y = np.meshgrid(X_range, X_range)
_x, _y = np.meshgrid(Y, Y)
a = 0.1
print(np.array(Y).shape)
print(Y)
def axes():
    plt.axhline(0,xmin=-4, xmax=4, alpha=.1)
    plt.axvline(0,ymin=-4, ymax=4, alpha=.1)

axes()
plt.contour(y, x, (_y**2 - 4*a*_x), [0], colors='k')
plt.show()
# result = probability_pr(X, )
# for i in range(len(X_range)):
#     y = ((-(0.5)*(X[i]-mean_class1).T)*np.linalg.inv(covariance_class1)*(X[i]-mean_class1) - (np.abs(100)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance_class1)) + np.log(prior1) )* 0.025
#     y1.append(np.linalg.det(y))
#     y = ((-(0.5)*(X[i]-mean_class2).T)*np.linalg.inv(covariance_class2)*(X[i]-mean_class2) - (np.abs(100)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance_class2)) + np.log(prior1) )* 0.025
#     y2.append(np.linalg.det(y))
