# กำหนดค่า
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm

# กำหนดค่าพารามิเตอร์ของการแจกแจง
mu_c1 = np.array([-1, -1])
mu_c2 = np.array([1, 1])
Sigma = np.array([[1, 0], [0, 1]])

# กำหนดค่าพารามิเตอร์ของการแจกแจง 2 มิติ
Mu_c1 = -1
Mu_c2 = 1
sigma = 1

# สร้างฟังก์ชัน Linear Discriminant
def linear_discriminant(mu_c1, mu_c2, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    w = np.dot(Sigma_inv, mu_c1 - mu_c2)
    b = -0.5 * np.dot(np.dot(mu_c1, Sigma_inv), mu_c1) + 0.5 * np.dot(np.dot(mu_c2, Sigma_inv), mu_c2)
    return w, b
# 2 มิติ
def linear_discriminant_1(mu_c1, mu_c2, sigma):
    W = (Mu_c2 - Mu_c1) / sigma**2
    B = -0.5 * (Mu_c1**2 - Mu_c2**2) / sigma**2
    return W, B

# คำนวณ w และ b
w, b = linear_discriminant(mu_c1, mu_c2, Sigma)
# W, B = linear_discriminant_1(Mu_c1, Mu_c2, sigma)

# สร้างกราฟ
x1_range = np.linspace(-4, 4, 400)
x2_range = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
pos = np.dstack((X1, X2))
x = np.linspace(-4, 4, 400)

# คำนวณ likelihood
rv_c1 = multivariate_normal(mu_c1, Sigma)
rv_c2 = multivariate_normal(mu_c2, Sigma)
likelihood_c1 = rv_c1.pdf(pos)
likelihood_c2 = rv_c2.pdf(pos)
# likelihood_c1_1 = norm.pdf(x, Mu_c1, sigma)
# likelihood_c2_1 = norm.pdf(x, Mu_c2, sigma)

# คำนวณ posterior โดยสมมติว่า p(c1) = p(c2) = 0.5
posterior_c1 = likelihood_c1 / (likelihood_c1 + likelihood_c2)
posterior_c2 = likelihood_c2 / (likelihood_c1 + likelihood_c2)
# posterior_c1_1 = likelihood_c1_1 / (likelihood_c1_1 + likelihood_c2_1)
# posterior_c2_1 = likelihood_c2_1 / (likelihood_c1_1 + likelihood_c2_1)

# คำนวณขอบตัดสินใจ
decision_boundary = w[0] * X1 + w[1] * X2 + b
# decision_boundary_sample = -B / W

# วาดกราฟ likelihood
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X1, X2, likelihood_c1, cmap='Reds', alpha=0.5)
plt.contourf(X1, X2, likelihood_c2, cmap='Greens', alpha=0.5)
# plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
plt.title('Likelihood Class 1 Red, Class 2 Green')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

# plt.subplot(1, 3, 1)
# plt.plot(x, likelihood_c1_1, 'k-', label='Class $c_1$')
# plt.plot(x, likelihood_c2_1, 'k--', label='Class $c_2$')
# # plt.axvline(decision_boundary_sample, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
# plt.title('Likelihood')
# plt.xlabel('$x$', fontsize=14)
# plt.ylabel('Likelihood', fontsize=14)
# plt.legend()

# plt.subplot(1, 3, 2)
# plt.contourf(X1, X2, likelihood_c2, cmap='Greens')
# plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
# plt.title('Likelihood Class 2')
# plt.xlabel(r'$x_1$', fontsize=14)
# plt.ylabel(r'$x_2$', fontsize=14)

# # วาดกราฟ posterior
# plt.subplot(1, 3, 2)
# plt.plot(x, posterior_c1_1, 'k-', label='Class $c_1$')
# plt.plot(x, posterior_c2_1, 'k--', label='Class $c_2$')
# # plt.axvline(decision_boundary_sample, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
# plt.title('Posterior Probability')
# plt.xlabel('$x$', fontsize=14)
# plt.ylabel('Posterior Probability', fontsize=14)
# plt.legend()

# วาดกราฟ posterior Linear Discriminant
plt.subplot(1, 2, 2)
plt.contourf(X1, X2, posterior_c1, cmap='Reds', alpha=0.5)
plt.contourf(X1, X2, posterior_c2, cmap='Greens', alpha=0.5)
plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
plt.title('Linear Discriminant')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

plt.tight_layout()
plt.show()
