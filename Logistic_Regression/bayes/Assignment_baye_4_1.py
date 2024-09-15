# กำหนดค่า
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# กำหนดค่าพารามิเตอร์ของการแจกแจง
mu_c1 = np.array([-1, -1])
mu_c2 = np.array([1, 1])
Sigma = np.array([[1, 0], [0, 1]])

# สร้างฟังก์ชัน Linear Discriminant
def linear_discriminant(mu_c1, mu_c2, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    w = np.dot(Sigma_inv, mu_c1 - mu_c2)
    b = -0.5 * np.dot(np.dot(mu_c1, Sigma_inv), mu_c1) + 0.5 * np.dot(np.dot(mu_c2, Sigma_inv), mu_c2)
    return w, b

# คำนวณ w และ b
w, b = linear_discriminant(mu_c1, mu_c2, Sigma)

# สร้างกราฟ
x1_range = np.linspace(-4, 4, 400)
x2_range = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
pos = np.dstack((X1, X2))

# คำนวณ likelihood
rv_c1 = multivariate_normal(mu_c1, Sigma)
rv_c2 = multivariate_normal(mu_c2, Sigma)
likelihood_c1 = rv_c1.pdf(pos)
likelihood_c2 = rv_c2.pdf(pos)

# คำนวณ posterior โดยสมมติว่า p(c1) = p(c2) = 0.5
posterior_c1 = likelihood_c1 / (likelihood_c1 + likelihood_c2)
posterior_c2 = likelihood_c2 / (likelihood_c1 + likelihood_c2)

# คำนวณขอบตัดสินใจ
decision_boundary = w[0] * X1 + w[1] * X2 + b

# วาดกราฟ likelihood
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X1, X2, likelihood_c1, cmap='Reds', alpha=0.5)
plt.contourf(X1, X2, likelihood_c2, cmap='Greens', alpha=0.5)
plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
plt.title('Likelihood Class 1 Red, Class 2 Green')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

# plt.subplot(1, 3, 2)
# plt.contourf(X1, X2, likelihood_c2, cmap='Greens')
# plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
# plt.title('Likelihood Class 2')
# plt.xlabel(r'$x_1$', fontsize=14)
# plt.ylabel(r'$x_2$', fontsize=14)

# วาดกราฟ posterior
plt.subplot(1, 2, 2)
plt.contourf(X1, X2, posterior_c1, cmap='Reds', alpha=0.5)
plt.contourf(X1, X2, posterior_c2, cmap='Greens', alpha=0.5)
plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
plt.title('Posterior Probabilities')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

plt.tight_layout()
plt.show()