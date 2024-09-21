import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 
from sklearn.datasets import make_circles, make_classification, make_moons
# from numpy import random as rd

# X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
np.random.seed(42)
X = np.random.rand(1, 100, 2)
print(X)
# print(np.array(X).reshape(-1, 2))
X = np.array(X).reshape(-1, 2)
y = np.random.choice([0,1],size=100)
# print(len(y))
print(y)
print(X[1][0])
# print(X[:,0])
# print(X[:,1])
for i in range(100):
    if y[i] == 1:
        plt.scatter(X[i][0], X[i][1], c='red')
    elif y[i] == 0:
        plt.scatter(X[i][0], X[i][1], c='blue')
plt.show()
# x_range = np.linspace(-4, 4, 100)
# x1_grid, x2_grid = np.meshgrid(x_range, x_range)

# means = [np.mean(X[y == k], axis=0) for k in np.unique(y)]
# covariances = [np.cov(X[y == k].T) for k in np.unique(y)]
# priors = [np.mean(y == k) for k in np.unique(y)]
# print(means)
# print(covariances)
# print(priors)
# def pooled_covariance(X, y):

#     classes = np.unique(y)
#     n_features = X.shape[1]
#     pooled_cov = np.zeros((n_features, n_features))
#     n_total = 0

#     for c in classes:
#         X_c = X[y == c]
#         n_c = X_c.shape[0]
#         cov_c = np.cov(X_c.T)
#         pooled_cov += (n_c - 1) * cov_c 
#         n_total += n_c - 1

#     return pooled_cov / n_total
# cov = pooled_covariance(X, y)

# def Sigmoid(z):
#     return 1/(1+np.exp(-z))

# def probability_density(x, mean, variance):
#     return (1/np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) **  2))/(2* variance)

# def qda_discriminant(x, mean, inv_cov, prior):
#     diff = x - mean
#     return -0.5 * np.dot(np.dot(diff.T, inv_cov), diff) + np.log(prior)

# def quadratic_discriminant(X, mean, prior, covariance):
#     inv_covariance = np.linalg.inv(covariance)
#     log_det_cov = np.log(np.linalg.det(covariance))
#     diff = X-mean
#     -0.5*((diff.T)*inv_covariance*diff) - 0.5*log_det_cov + np.log(prior)
#     return y

# def probability_pr(X):
#     discriminants = np.array([
#         [qda_discriminant(x, means[k], covariances[k], priors[k]) for k in range(len(means))]
#         for x in X
#     ])
#     return np.argmax(discriminants, axis=1)

# def qda_posterior(X):
#     discriminants = probability_pr(X)
#     max_discriminants = np.max(discriminants, axis=1, keepdims=True)
#     exp_discriminants = np.exp(discriminants - max_discriminants)
#     posteriors = exp_discriminants / np.sum(exp_discriminants, axis=1, keepdims=True)
#     return posteriors

# mean_class_1 = np.array([2, 0])
# mean_class_2 = np.array([1, 0])
# cov_class_1 = np.array([[0.5, 0],[0, 0.25]])
# cov_class_2 = np.array([[0.25, 0],[0, 0.5]])
# inv_cov_class1 = np.linalg.inv(cov_class_1)
# inv_cov_class2 = np.linalg.inv(cov_class_2)
# prior_class_1 = 0.5
# prior_class_2 = 0.5


# plt.title("Quadratic Discriminant Analysis (QDA), Decision Boundary")
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=100, cmap='viridis')
# plt.scatter([mean[0] for mean in means], [mean[1] for mean in means], c='red', marker='x')

# plt.show()
# Gaussian contours
# rv_class_1 = multivariate_normal(mean_class_1, cov_class_1)
# rv_class_2 = multivariate_normal(mean_class_2, cov_class_2)
# x, y = np.meshgrid(x_range, x_range)
# pos = np.dstack((x, y))
# plt.contour(x, y, , levels=[0.1, 0.2, 0.3], cmap='viridis')
# plt.contour(x, y, rv_class_1.pdf(pos), levels=[0.1, 0.2, 0.3], colors='blue', linestyles='dashed')
# plt.contour(x, y, rv_class_2.pdf(pos), levels=[0.1, 0.2, 0.3], colors='red', linestyles='dashed')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.grid(True)
# plt.show()