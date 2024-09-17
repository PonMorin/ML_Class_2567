import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# กำหนดค่าพารามิเตอร์ของการแจกแจง
mu_c1 = np.array([-1, -1])
mu_c2 = np.array([1, 1])
Sigma = np.array([[1, 0], [0, 1]])

# กำหนดค่า prior
prior_c1 = 0.7  # Prior probability of class 1
prior_c2 = 0.3  # Prior probability of class 2

# สร้างฟังก์ชัน Linear Discriminant
def linear_discriminant(mu_c1, mu_c2, Sigma, prior_c1, prior_c2):
    Sigma_inv = np.linalg.inv(Sigma)
    w = np.dot(Sigma_inv, mu_c1 - mu_c2)
    b = -0.5 * np.dot(np.dot(mu_c1, Sigma_inv), mu_c1) + 0.5 * np.dot(np.dot(mu_c2, Sigma_inv), mu_c2)
    b += np.log(prior_c1 / prior_c2)  # เพิ่มผลของ prior probabilities
    return w, b

# คำนวณ w และ b
w, b = linear_discriminant(mu_c1, mu_c2, Sigma, prior_c1, prior_c2)

# สร้างกราฟ
x1_range = np.linspace(-4, 4, 400)
x2_range = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
posL = np.dstack((X1, X2))
x = np.linspace(-4, 4, 400)

# คำนวณ likelihood
rv_c1 = multivariate_normal(mu_c1, Sigma)
rv_c2 = multivariate_normal(mu_c2, Sigma)

# สำหรับ 2 มิติ
likelihood_c1L = rv_c1.pdf(posL)
likelihood_c2L = rv_c2.pdf(posL)

# สำหรับ 1 มิติ เราต้องการ fix หนึ่งใน feature ที่ค่าใดค่าหนึ่ง เช่น x2 = 0
x2_fixed = 0
pos = np.column_stack((x1_range, np.full_like(x1_range, x2_fixed)))

likelihood_c1 = rv_c1.pdf(pos)
likelihood_c2 = rv_c2.pdf(pos)

# คำนวณ posterior โดยใช้ Bayes' theorem
posterior_c1L = prior_c1 * likelihood_c1L / (prior_c1 * likelihood_c1L + prior_c2 * likelihood_c2L)
posterior_c2L = prior_c2 * likelihood_c2L / (prior_c1 * likelihood_c1L + prior_c2 * likelihood_c2L)

# คำนวณขอบตัดสินใจ
decision_boundary = w[0] * X1 + w[1] * X2 + b

# สร้างกราฟ likelihood และ posterior
plt.figure(figsize=(12, 12))

# วาดกราฟ likelihood สำหรับ 1 มิติ
plt.subplot(2, 2, 1)
plt.plot(x1_range, likelihood_c1, label='Class 1', color='red')
plt.plot(x1_range, likelihood_c2, label='Class 2', color='green')
plt.title('1D Likelihood')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel('Likelihood', fontsize=14)
plt.legend()

# วาดกราฟ posterior สำหรับ 1 มิติ
plt.subplot(2, 2, 2)
plt.plot(x1_range, posterior_c1L[:, int(len(x2_range)/2)], label='Class 1', color='red')
plt.plot(x1_range, posterior_c2L[:, int(len(x2_range)/2)], label='Class 2', color='green')
plt.title('1D Posterior')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel('Posterior', fontsize=14)
plt.legend()

# วาดกราฟ likelihood สำหรับ 2 มิติ
plt.subplot(2, 2, 3)
plt.contourf(X1, X2, likelihood_c1L, cmap='Reds', alpha=0.5)
plt.contourf(X1, X2, likelihood_c2L, cmap='Greens', alpha=0.5)
plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
plt.title('2D Likelihood')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

# วาดกราฟ Linear Discriminant สำหรับ 2 มิติ
plt.subplot(2, 2, 4)
plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
plt.title('2D Linear Discriminant')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)

plt.tight_layout()
plt.show()
