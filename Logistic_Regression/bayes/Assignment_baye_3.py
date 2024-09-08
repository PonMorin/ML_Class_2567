# 3.เขียนโปรแกรมสำหรับสร้างตัวจำแนกกำลังสอง
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
# from scipy.optimize import minimize
# from scipy.special import logsumexp
# from scipy.stats import poisson
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def standard_deviation(n, X):
    mean = np.sum(X)/n
    variance = sum([((x-mean)**2) for x in X])/(n-1)
    result = variance ** 0.5
    return result

def density_function(mean, X, std):
    # print(mean)
    # return (1/(np.sqrt(2*np.pi))*std)*np.exp(np.negative((X - mean)**2/(2*std**2)))
    return np.exp(np.negative((X - mean)**2/(2*std**2)))/((np.sqrt(2*np.pi))*std)

def normal_distribution(n, X, std):
    mean = np.sum(X)/n
    variance = sum([((x-mean)**2) for x in X])/n
    std = variance ** 0.5
    return np.exp(np.negative(0.5*(np.square((X - mean)/std))))/((np.sqrt(2*np.pi))*std)

def single_threshold(n, X, Class):
    mean = np.sum(X)/n
    variance = sum([((x-mean)**2) for x in X])/n
    std = variance ** 0.5
    # print(X-mean)
    return np.negative(np.square(X - mean)/2*variance) - 0.5*np.log(2*np.pi) - np.log(std) - np.log(normal_distribution(len(Class), Class, np.std(Class))) - np.log(normal_distribution(len(X), X, np.std(X)))

def linear_discriminant(mu_c1, mu_c2, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    print(Sigma_inv)
    w = np.dot(Sigma_inv, mu_c1 - mu_c2)
    print(w)
    b = -0.5 * np.dot(np.dot(mu_c1, Sigma_inv), mu_c1) + 0.5 * np.dot(np.dot(mu_c2, Sigma_inv), mu_c2)
    print(b)
    return w, b

def quadratic_g(X, mean, D, sigma_class):
    return np.exp(np.negative((X-mean).T)*sum(X-mean))/(((2*np.pi)**(D/2))*np.abs(sigma_class)**0.5)

# def quadratic_discriminant(X, mean, priors, attribute, sigma_class):
#     n_class = priors
#     np.negative(0.5)*((X-mean).T)*np.linalg.inv(sigma_class) - (attribute/2)*np.log(2*np.pi) - 0.5*np.log(abs(sigma_class)) + np.log(quadratic_g(c)) - np.log(quadratic_g())


# X = [1, 2]
# mean_class1 = [[1, 0]]
# mean_class2 = [[2, 0]]
# class1 = [[0.5, 0],[0, 0.25]]
# class2 = [[0.25, 0],[0, 0.5]]

# np.random.seed(42)
samples_c1 = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 100)
samples_c2 = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 100)
c1 =  np.arange(500)
c2 =  np.arange(500)

mu_c1_sample = np.mean(samples_c1, axis=0)
mu_c2_sample = np.mean(samples_c2, axis=0)
Sigma_sample = np.cov(np.vstack((samples_c1, samples_c2)).T)
Sigma = np.cov(np.vstack((c1, c2)).T)
# print(Sigma_sample)

X1_range = np.linspace(-4, 4, 500)
X2_range = np.linspace(-4, 4, 500)
X1, X2 = np.meshgrid(X1_range, X2_range)
print(X1)
y = quadratic_g(X1, np.mean(X1), 2, Sigma)

plt.figure(figsize=(8, 8))

# plt.contour(X1, X2, decision_boundary_sample, levels=[0], colors='black')
# plt.scatter(samples_c1[:, 0], samples_c1[:, 1], c='blue', marker='o', label='Class 1')
# plt.scatter(samples_c2[:, 0], samples_c2[:, 1], c='green', marker='s', label='Class 2')
plt.plot(X1, y)
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.title('', fontsize=16, fontname='Tahoma')
plt.legend()
plt.grid(True)
plt.show()