# สุ่มค่า
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# สุ่มตัวอย่างจากการแจกแจง
np.random.seed(42)  # กำหนด seed เพื่อให้ผลลัพธ์การสุ่มซ้ำได้
samples_c1 = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 100)
samples_c2 = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 100)
# สุ่มตัวอย่างจากการแจกแจง 1 มิติ
np.random.seed(42)  # กำหนด seed เพื่อให้ผลลัพธ์การสุ่มซ้ำได้
Samples_c1 = np.random.normal(-1, 1, 100)
Samples_c2 = np.random.normal(1, 1, 100)

# คำนวณค่า mean และ covariance จากตัวอย่างที่สุ่มได้
mu_c1_sample = np.mean(samples_c1, axis=0)
mu_c2_sample = np.mean(samples_c2, axis=0)
Sigma_sample = np.cov(np.vstack((samples_c1, samples_c2)).T)
# คำนวณค่า mean และ variance จากตัวอย่างที่สุ่มได้
Mu_c1_sample = np.mean(Samples_c1)
Mu_c2_sample = np.mean(Samples_c2)
sigma_sample = np.std(np.concatenate((Samples_c1, Samples_c2)))

# สร้างฟังก์ชัน Linear Discriminant
def linear_discriminant(mu_c1, mu_c2, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    w = np.dot(Sigma_inv, mu_c1 - mu_c2)
    b = -0.5 * np.dot(np.dot(mu_c1, Sigma_inv), mu_c1) + 0.5 * np.dot(np.dot(mu_c2, Sigma_inv), mu_c2)
    return w, b

# คำนวณ w และ b จากตัวอย่างที่สุ่มได้
w_sample, b_sample = linear_discriminant(mu_c1_sample, mu_c2_sample, Sigma_sample)

# สร้างกราฟ
x1_range = np.linspace(-4, 4, 400)
x2_range = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
pos = np.dstack((X1, X2))
x = np.linspace(-4, 4, 400)

# คำนวณ likelihood
rv_c1_sample = multivariate_normal(mu_c1_sample, Sigma_sample)
rv_c2_sample = multivariate_normal(mu_c2_sample, Sigma_sample)
likelihood_c1_sample = rv_c1_sample.pdf(pos)
likelihood_c2_sample = rv_c2_sample.pdf(pos)

# คำนวณ posterior โดยสมมติว่า p(c1) = p(c2) = 0.5
# posterior_c1_sample = likelihood_c1_sample / (likelihood_c1_sample + likelihood_c2_sample)
# posterior_c2_sample = likelihood_c2_sample / (likelihood_c1_sample + likelihood_c2_sample)
posterior_c1_sample = 0.5
posterior_c2_sample = 0.5

# คำนวณขอบตัดสินใจ
decision_boundary_sample = w_sample[0] * X1 + w_sample[1] * X2 + b_sample

# วาดกราฟ likelihood
plt.figure(figsize=(6, 6))

# วาดกราฟ posterior
# plt.subplot(1, 2, 1)
# plt.contourf(X1, X2, posterior_c1_sample, cmap='Reds', alpha=0.5)
# plt.contourf(X1, X2, posterior_c2_sample, cmap='Greens', alpha=0.5)
plt.contour(X1, X2, decision_boundary_sample, levels=[0], colors='black')
# plt.scatter(samples_c1[:, 0], samples_c1[:, 1], c='red', marker='o', label='Class 1')
# plt.scatter(samples_c2[:, 0], samples_c2[:, 1], c='green', marker='s', label='Class 2')
plt.title('Posterior Probabilities')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.legend()

# plt.subplot(1, 2, 2)
# plt.contourf(X1, X2, likelihood_c1_sample, cmap='Reds', alpha=0.5)
# plt.contourf(X1, X2, likelihood_c2_sample, cmap='Greens', alpha=0.5)
# plt.title('Likelihood Class 1, Class 2')
# plt.xlabel(r'$x_1$', fontsize=14)
# plt.ylabel(r'$x_2$', fontsize=14)

plt.tight_layout()
plt.show()
