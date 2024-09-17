# สุ่มค่า
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# กำหนดค่าพารามิเตอร์ของการแจกแจง
mu_c1 = np.array([-1, -1])
mu_c2 = np.array([1, 1])
Sigma = np.array([[1, 0], [0, 1]])

# สุ่มตัวอย่างข้อมูล
n_samples = 1000
samples_c1 = multivariate_normal.rvs(mean=mu_c1, cov=Sigma, size=n_samples)
samples_c2 = multivariate_normal.rvs(mean=mu_c2, cov=Sigma, size=n_samples)

# คำนวณค่าพารามิเตอร์จากตัวอย่าง
estimated_mu_c1 = np.mean(samples_c1, axis=0)
estimated_mu_c2 = np.mean(samples_c2, axis=0)
estimated_Sigma_c1 = np.cov(samples_c1, rowvar=False)
estimated_Sigma_c2 = np.cov(samples_c2, rowvar=False)

# สร้างฟังก์ชัน Linear Discriminant
def linear_discriminant(mu_c1, mu_c2, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    w = np.dot(Sigma_inv, mu_c1 - mu_c2)
    b = -0.5 * np.dot(np.dot(mu_c1, Sigma_inv), mu_c1) + 0.5 * np.dot(np.dot(mu_c2, Sigma_inv), mu_c2)
    return w, b

# คำนวณ w และ b โดยใช้ค่าพารามิเตอร์ที่ประมาณค่าได้
w, b = linear_discriminant(estimated_mu_c1, estimated_mu_c2, estimated_Sigma_c1)

# สร้างกราฟ
x1_range = np.linspace(-4, 4, 400)
x2_range = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
posL = np.dstack((X1, X2))
x = np.linspace(-4, 4, 400)

# คำนวณ likelihood สำหรับการแจกแจงที่ประมาณค่าได้
rv_c1 = multivariate_normal(estimated_mu_c1, estimated_Sigma_c1)
rv_c2 = multivariate_normal(estimated_mu_c2, estimated_Sigma_c2)

# สำหรับ 2 มิติ
likelihood_c1L = rv_c1.pdf(posL)
likelihood_c2L = rv_c2.pdf(posL)

# สำหรับ 1 มิติ เราต้องการ fix หนึ่งใน feature ที่ค่าใดค่าหนึ่ง เช่น x2 = 0
x2_fixed = 0
pos = np.column_stack((x1_range, np.full_like(x1_range, x2_fixed)))

likelihood_c1 = rv_c1.pdf(pos)
likelihood_c2 = rv_c2.pdf(pos)

# คำนวณ posterior โดยสมมติว่า p(c1) = p(c2) = 0.5
posterior_c1 = likelihood_c1 / (likelihood_c1 + likelihood_c2)
posterior_c2 = likelihood_c2 / (likelihood_c1 + likelihood_c2)

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
plt.plot(x1_range, posterior_c1, label='Class 1', color='red')
plt.plot(x1_range, posterior_c2, label='Class 2', color='green')
plt.title('1D Posterior')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel('Posterior', fontsize=14)
plt.legend()

# วาดกราฟ likelihood สำหรับ 2 มิติ
plt.subplot(2, 2, 3)
plt.contourf(X1, X2, likelihood_c1L, cmap='Reds', alpha=0.5)
plt.contourf(X1, X2, likelihood_c2L, cmap='Greens', alpha=0.5)
plt.scatter(samples_c1[:, 0], samples_c1[:, 1], c='red', marker='o', label='Class 1', alpha=0.01)
plt.scatter(samples_c2[:, 0], samples_c2[:, 1], c='green', marker='s', label='Class 2', alpha=0.01)
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