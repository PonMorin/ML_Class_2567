import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 

def probability_density(x, mean, variance):
    return (1/np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) **  2))/(2* variance)

mean_class_1 = np.array([2, 0])
mean_class_2 = np.array([1, 0])
cov_class_1 = np.array([[0.5, 0],[0, 0.25]])
cov_class_2 = np.array([[0.25, 0],[0, 0.5]])
inv_cov_class1 = np.linalg.inv(cov_class_1)
inv_cov_class2 = np.linalg.inv(cov_class_2)
prior_class_1 = 0.5
prior_class_2 = 0.5

def qda_discriminant(x, mean, inv_cov, prior):
    diff = x - mean
    return -0.5 * np.dot(np.dot(diff.T, inv_cov), diff) + np.log(prior)

x_range = np.linspace(-4, 4, 100)
x1_grid, x2_grid = np.meshgrid(x_range, x_range)

g1_values = np.zeros_like(x1_grid)
g2_values = np.zeros_like(x2_grid)
for i in range(x1_grid.shape[0]):
    for j in range(x2_grid.shape[0]):
        x = np.array([x1_grid[i, j],x2_grid[i, j]])
        g1_values[i, j] = qda_discriminant(x, mean_class_1, inv_cov_class1, prior_class_1)
        g2_values[i, j] = qda_discriminant(x, mean_class_2, inv_cov_class2, prior_class_2)

decision_boundary = np.abs(g1_values - g2_values).argmin()


plt.figure(figsize=(8, 6))
plt.contour(x1_grid, x2_grid, g1_values - g2_values, levels=[0], colors='black')
plt.title("Quadratic Discriminant Analysis (QDA), Decision Boundary")
# Gaussian contours
rv_class_1 = multivariate_normal(mean_class_1, cov_class_1)
rv_class_2 = multivariate_normal(mean_class_2, cov_class_2)
x, y = np.meshgrid(x_range, x_range)
pos = np.dstack((x, y))
plt.contour(x, y, rv_class_1.pdf(pos), levels=[0.1, 0.2, 0.3], colors='blue', linestyles='dashed')
plt.contour(x, y, rv_class_2.pdf(pos), levels=[0.1, 0.2, 0.3], colors='red', linestyles='dashed')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.show()

# np.random.seed(42)

# plt.figure(figsize=(8, 6))
# plt.contour(x1_grid, x2_grid, g1_values - g2_values, levels=[0], color='black')
# plt.title("Quadratic Discriminant Analysis (QDA), Decision Boundary")

# # Gaussian contours
# rv_class_1 = multivariate_normal(mean_class_1_sample, cov_class_1)
# rv_class_2 = multivariate_normal(mean_class_2_sample, cov_class_2)
# x, y = np.meshgrid(x_range, x_range)
# pos = np.dstack((x, y))
# plt.contour(x, y, rv_class_1.pdf(pos), levels=[0.1, 0.2, 0.3], colors='blue', linestyles='dashed')
# plt.contour(x, y, rv_class_2.pdf(pos), levels=[0.1, 0.2, 0.3], colors='red', linestyles='dashed')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.grid(True)
# plt.show()

# plt.scatter(sample_class_1[:, 0]. sample_class_1[:, 1])

# plt.figure(figsize=(8, 6))
# plt.title("QDA Decision Boundary with Sampled Data")
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.grid(True)

# plt.scatter(samples_class_1[:, 0], samples_class_1[:, 1])