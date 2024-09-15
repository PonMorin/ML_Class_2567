
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
import matplotlib as mpl

def Sigmoid(z):
    return 1/(1+np.exp(-z))

def density_function(mean, X, std):
    # return (1/(np.sqrt(2*np.pi))*std)*np.exp(np.negative((X - mean)**2/(2*std**2)))
    return np.exp(np.negative((X - mean)**2/(2*std**2)))/((np.sqrt(2*np.pi))*std)

# def probability_pr(X, mean, D, covariance):
#     return np.exp(np.negative(0.5)*(X-mean).T*np.linalg.inv(sum(X-mean)))/(((2*np.pi)**(D/2))*np.abs(covariance)**0.5)

def probability_pr(X, mean, D, covariance):
    history = []
    top = np.exp(-0.5*((X - mean).T)*1/np.sum(X - mean))
    down = ((2*np.pi)**(D/2))*np.abs(covariance)**0.5
    for i in range(len(X)):
        history.append(top.T[i]/np.linalg.det(down))
        # print(top.T[i]/np.linalg.det(down))
    return history

def quadratic_discriminant(X, mean, D, prior, covariance):
    y = []
    for i in range(len(X)):
        _y = ((-(0.5)*(X[i]-mean).T)*np.linalg.inv(covariance)*(X[i]-mean) - (np.abs(D)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance)) + np.log(prior)) * 0.025 - np.sum(probability_pr(X, mean, D, covariance))
        # _y = ((-(0.5)*(X[i]-mean).T)*(1/(covariance)*(X[i]-mean))) - (np.abs(D)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance)) + np.log(prior) * 0.025 - np.sum(probability_pr(X, mean, D, covariance))
        y.append(np.linalg.det(_y))
        # y.append(_y)

    return y
    
    # return np.exp(np.negative(0.5)*(X-mean).T*np.linalg.inv(sum(X-mean)))/(((2*np.pi)**(D/2))*np.abs(covariance)**0.5)

# def quadratic_discriminant(X, mean, attribute, sigma_class):
#     # return np.negative(0.5)*((X-mean).T)*np.linalg.inv(sigma_class)-(np.abs(attribute)/2)*np.log(2*np.pi) - 0.5*np.log(abs(sigma_class)) + np.log(quadratic_g(attribute, np.mean(attribute), len(attribute), sigma_class))-np.log(quadratic_g(X, np.mean(X), len(X), sigma_class))
#     A = np.negative(0.5)*((X-mean).T)*np.linalg.inv(sigma_class)
#     B = (np.abs(attribute)/2)*np.log(2*np.pi)
#     C = 0.5*np.log(abs(sigma_class))
#     D = np.log(probability_pr(attribute, np.mean(attribute), len(attribute), sigma_class))
#     E = np.log(probability_pr(X, np.mean(X), len(X), sigma_class))

    # return A - B - C  + D - E


# μ1 = np.array([2, -1])
# μ2 = np.array([1, -1.5])
μ1 = np.array([2, 0])
μ2 = np.array([1, 0])
std1 = 1
std2 = 1
attribute = 100
X_range = np.linspace(-4, 4, attribute)
# covariance_class1 = [[0.5, 0],[0, 0.25]]
# covariance_class2 = [[0.25, 0],[0, 0.5]]
covariance_class1 = [[2, 0],[0, 4]]
covariance_class2 = [[4, 0],[0, 2]]
likelihood1 = norm.pdf(X_range, μ1[0], std1)
likelihood2 = norm.pdf(X_range, μ2[0], std2)
prior1 = 0.5
prior2 = 0.5

# Calculate the posterior probabilities using Bayes' Rule
posterior1 = likelihood1 * prior1
posterior2 = likelihood2 * prior2

# Normalize the posteriors (since we're assuming equal priors)
total_posterior = posterior1 + posterior2
posterior1 /= total_posterior
posterior2 /= total_posterior

# Plot the likelihoods (Top Plot)
plt.figure(figsize=(6, 10))

plt.subplot(2, 1, 1)
plt.plot(X_range, likelihood1, label='Class $c_1$', color='black', linestyle='-')
plt.plot(X_range, likelihood2, label='Class $c_2$', color='black', linestyle='--')
plt.title('Likelihood')
plt.xlabel('x')
plt.ylabel('Likelihood')
plt.legend()
plt.grid(True)

# Plot the posterior probabilities (Bottom Plot)
plt.subplot(2, 1, 2)
plt.plot(X_range, posterior1, label='Class $c_1$', color='black', linestyle='-')
plt.plot(X_range, posterior2, label='Class $c_2$', color='black', linestyle='--')
plt.title('Posterior Probability')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

X = np.stack((X_range, X_range), axis=1)
Y1 = quadratic_discriminant(X, μ1, attribute, prior1, covariance_class1)
Y2 = quadratic_discriminant(X, μ2, attribute, prior2, covariance_class2)
likelihood1 = quadratic_discriminant(X, μ1, attribute, prior1, covariance_class1)
likelihood2 = quadratic_discriminant(X, μ2, attribute, prior2, covariance_class2)
evidence1 = probability_pr(X, μ1, attribute, covariance_class1)
evidence2 = probability_pr(X, μ2, attribute, covariance_class2)
posterior1 = []
posterior2 = []
for i in range(attribute):
    posterior1.append(np.mean(np.array(likelihood2[i])*prior1/evidence1[i]))
    posterior2.append(np.mean(np.array(likelihood2[i])*prior1/evidence2[i]))


x, y = np.meshgrid(X_range, X_range)


# plt.plot(X_range, posterior1)
# plt.plot(X_range, posterior2)
# plt.show()
# _x, _y = np.meshgrid(Y2, Y1)
# _x, _y = np.meshgrid(posterior2, posterior1)

plt.plot(posterior1, X_range)
plt.plot(np.negative(posterior2), X_range)
# a = 0.1

# def axes():
#     plt.axhline(0,xmin=-4, xmax=4, alpha=.1)
#     plt.axvline(0,ymin=-4, ymax=4, alpha=.1)

# axes()


# plt.contour(y, x, (_y**2 - 4*a*_x), [0], colors='k')
# plt.contour(x, y, (_y**2 - 4*a*_x), [0], colors='k')
# plt.contour(x, y, posterior, [0], colors='k')
plt.grid(True)
plt.show()
