# 3.เขียนโปรแกรมสำหรับสร้างตัวจำแนกกำลังสอง
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm

def density_function(mean, X, std):
    # return (1/(np.sqrt(2*np.pi))*std)*np.exp(np.negative((X - mean)**2/(2*std**2)))
    return np.exp(np.negative((X - mean)**2/(2*std**2)))/((np.sqrt(2*np.pi))*std)

# def probability_pr(X, mean, D, covariance):
#     return np.exp(np.negative(0.5)*(X-mean).T*np.linalg.inv(sum(X-mean)))/(((2*np.pi)**(D/2))*np.abs(covariance)**0.5)

# def quadratic_discriminant(X, mean, attribute, sigma_class):
#     # return np.negative(0.5)*((X-mean).T)*np.linalg.inv(sigma_class)-(np.abs(attribute)/2)*np.log(2*np.pi) - 0.5*np.log(abs(sigma_class)) + np.log(quadratic_g(attribute, np.mean(attribute), len(attribute), sigma_class))-np.log(quadratic_g(X, np.mean(X), len(X), sigma_class))
#     A = np.negative(0.5)*((X-mean).T)*np.linalg.inv(sigma_class)
#     B = (np.abs(attribute)/2)*np.log(2*np.pi)
#     C = 0.5*np.log(abs(sigma_class))
#     D = np.log(probability_pr(attribute, np.mean(attribute), len(attribute), sigma_class))
#     E = np.log(probability_pr(X, np.mean(X), len(X), sigma_class))
    # return A - B - C  + D - E

# X = [1, 2]
mean_class1 = np.array([2, 0])
mean_class2 = np.array([1, 0])
X_range = np.linspace(-4, 4, 100)

covariance_class1 = [[0.5, 0],[0, 0.25]]
covariance_class2 = [[0.25, 0],[0, 0.5]]
mean1, std1 = 2, 1
mean2, std2 = 1, 1
attribute = 100
X = np.stack((X_range, X_range), axis=1)
# print(np.linalg.inv(mean_class1))
# print(sum(X-mean_class1))

# probability_pr = np.exp(np.negative(0.5)*(X-mean).T*np.linalg.inv(sum(X-mean)))/(((2*np.pi)**(D/2))*np.abs(covariance_class1)**0.5)

# likelihood1 = probability_pr(X, mean1, ma, attribute, covariance_class1)
# likelihood2 = probability_pr(X, mean_class2, attribute, covariance_class2)
likelihood1 = norm.pdf(X_range, mean1, std1)
likelihood2 = norm.pdf(X_range, mean2, std2)

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

# Define prior probabilities (assumed equal priors)
prior1 = 0.5
prior2 = 0.5

# Calculate the posterior probabilities using Bayes' Rule
posterior1 = likelihood1 * prior1
posterior2 = likelihood2 * prior2

# Normalize the posteriors (since we're assuming equal priors)
total_posterior = posterior1 + posterior2
posterior1 /= total_posterior
posterior2 /= total_posterior

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

# evidence1 = quadratic_discriminant(X, mean1, attribute, covariance_class1)
# evidence2 = quadratic_discriminant(X, mean2, attribute, covariance_class2)
# posterior1 = (likelihood1 * prior1)/evidence1
# posterior2 = (likelihood2 * prior2)/evidence2

y1 = []
y2 = []
print(np.linalg.inv(covariance_class1))
for i in range(len(X_range)):
    y = ((-(0.5)*(X[i]-mean_class1).T)*np.linalg.inv(covariance_class1)*(X[i]-mean_class1) - (np.abs(100)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance_class1)) + np.log(prior1) )* 0.025
    y1.append(np.linalg.det(y))
    y = ((-(0.5)*(X[i]-mean_class2).T)*np.linalg.inv(covariance_class2)*(X[i]-mean_class2) - (np.abs(100)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(covariance_class2)) + np.log(prior1) )* 0.025
    y2.append(np.linalg.det(y))


# def axes():
#     plt.axhline(0, alpha=.1)
#     plt.axvline(0, alpha=.1)

# a = .3
# axes()
# print(y1)
plt.figure(figsize=(8, 8))
plt.plot(y1, X_range)
plt.plot(np.negative(y2), X_range)
print(np.stack((np.float_power(y2, 2), X_range), axis=1))
# plt.contour(X, y1, (np.float_power(y1, 2) - 4*a*X), [0], colors='k')
# plt.contour(-X, y2, (np.float_power(y2, 2) - 4*a*X), [0], colors='k')
# x,y = np.meshgrid(x, y)

# # plt.contour(X1, X2, decision_boundary_sample, levels=[0], colors='black')
# # plt.scatter(samples_c1[:, 0], samples_c1[:, 1], c='blue', marker='o', label='Class 1')
# # plt.scatter(samples_c2[:, 0], samples_c2[:, 1], c='green', marker='s', label='Class 2')
# plt.plot(X1, posterior1, label="Class $c_1$")
# plt.plot(X2, posterior2, label="Class $c_2$", linestyle='--')

plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.title('', fontsize=16, fontname='Tahoma')
plt.legend()
plt.grid(True)
plt.show()