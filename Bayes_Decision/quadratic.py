import numpy as np
import matplotlib.pyplot as plt

# Parameters for Class 1
mu1 = np.array([2, 0])
cov1 = np.array([[0.5, 0], [0, 0.25]])

# Parameters for Class 2
mu2 = np.array([1, 0])
cov2 = np.array([[0.25, 0], [0, 0.5]])

# Prior probabilities for both classes
prior1 = 0.5
prior2 = 0.5

# Define the quadratic discriminant function
def quadratic_discriminant(x, mu, cov, prior):
    cov_inv = np.linalg.inv(cov)
    term1 = -0.5 * np.dot(np.dot((x - mu).T, cov_inv), (x - mu))
    term2 = -0.5 * np.log(np.linalg.det(cov))
    term3 = np.log(prior)
    return term1 + term2 + term3

# Create a grid of points to evaluate the discriminant functions
x_min, x_max = -4, 4
y_min, y_max = -4, 4
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Convert to 2D points

# Compute discriminant values for both classes
g1_values = np.array([quadratic_discriminant(point, mu1, cov1, prior1) for point in grid_points])
g2_values = np.array([quadratic_discriminant(point, mu2, cov2, prior2) for point in grid_points])

# Reshape to match grid
g1_values = g1_values.reshape(xx.shape)
g2_values = g2_values.reshape(xx.shape)

# Decision boundary: where g1 = g2
decision_boundary = g1_values - g2_values

# Plot decision boundary
plt.contour(xx, yy, decision_boundary, levels=[0], cmap="RdBu", linewidths=2)

# Plot mean points for each class
plt.scatter(mu1[0], mu1[1], color='blue', label='Class 1 Mean')
plt.scatter(mu2[0], mu2[1], color='red', label='Class 2 Mean')

# Configure plot
plt.title("QDA Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.show()
