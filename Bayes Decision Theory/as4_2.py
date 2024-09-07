# สุ่มค่า
import numpy as np
import matplotlib.pyplot as plt

# สุ่มตัวอย่างจากการแจกแจง
np.random.seed(42)  # กำหนด seed เพื่อให้ผลลัพธ์การสุ่มซ้ำได้
samples_c1 = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 100)
samples_c2 = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 100)

# คำนวณค่า mean และ covariance จากตัวอย่างที่สุ่มได้
mu_c1_sample = np.mean(samples_c1, axis=0)
mu_c2_sample = np.mean(samples_c2, axis=0)
Sigma_sample = np.cov(np.vstack((samples_c1, samples_c2)).T)

# สร้างฟังก์ชัน Linear Discriminant (เหมือนเดิม)
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
decision_boundary_sample = w_sample[0] * X1 + w_sample[1] * X2 + b_sample

plt.figure(figsize=(8, 8))
plt.contour(X1, X2, decision_boundary_sample, levels=[0], colors='black')
plt.scatter(samples_c1[:, 0], samples_c1[:, 1], c='blue', marker='o', label='Class 1')
plt.scatter(samples_c2[:, 0], samples_c2[:, 1], c='green', marker='s', label='Class 2')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.title('ขอบตัดสินใจโดยใช้ตัวอย่างสุ่ม', fontsize=16, fontname='Tahoma')
plt.legend()
plt.grid(True)
plt.show()
