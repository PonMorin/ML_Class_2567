import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# สุ่มตัวอย่างจากการแจกแจง
np.random.seed(42)  # กำหนด seed เพื่อให้ผลลัพธ์การสุ่มซ้ำได้
samples_c1 = np.random.normal(-1, 1, 100)
samples_c2 = np.random.normal(1, 1, 100)

# คำนวณค่า mean และ variance จากตัวอย่างที่สุ่มได้
mu_c1_sample = np.mean(samples_c1)
mu_c2_sample = np.mean(samples_c2)
sigma_sample = np.std(np.concatenate((samples_c1, samples_c2)))

# สร้างฟังก์ชัน Linear Discriminant (ปรับสำหรับ 1 มิติ)
def linear_discriminant(mu_c1, mu_c2, sigma):
    w = (mu_c2 - mu_c1) / sigma**2
    b = -0.5 * (mu_c1**2 - mu_c2**2) / sigma**2
    return w, b

# คำนวณ w และ b จากตัวอย่างที่สุ่มได้
w_sample, b_sample = linear_discriminant(mu_c1_sample, mu_c2_sample, sigma_sample)

# สร้างกราฟ
x = np.linspace(-4, 4, 400)

# คำนวณ likelihood
likelihood_c1_sample = norm.pdf(x, mu_c1_sample, sigma_sample)
likelihood_c2_sample = norm.pdf(x, mu_c2_sample, sigma_sample)

# คำนวณ posterior โดยสมมติว่า p(c1) = p(c2) = 0.5
posterior_c1_sample = likelihood_c1_sample / (likelihood_c1_sample + likelihood_c2_sample)
posterior_c2_sample = likelihood_c2_sample / (likelihood_c1_sample + likelihood_c2_sample)

# คำนวณขอบตัดสินใจ
decision_boundary_sample = -b_sample / w_sample

# วาดกราฟ likelihood
plt.figure(figsize=(18, 6))

plt.subplot(2, 1, 1)
plt.plot(x, likelihood_c1_sample, 'k-', label='Class $c_1$')
plt.plot(x, likelihood_c2_sample, 'k--', label='Class $c_2$')
plt.axvline(decision_boundary_sample, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
# plt.scatter(samples_c1, np.zeros_like(samples_c1), c='blue', marker='o', label='Samples Class 1')
# plt.scatter(samples_c2, np.zeros_like(samples_c2), c='green', marker='s', label='Samples Class 2')
plt.title('Likelihood')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('Likelihood', fontsize=14)
plt.legend()

# วาดกราฟ posterior
plt.subplot(2, 1, 2)
plt.plot(x, posterior_c1_sample, 'k-', label='Class $c_1$')
plt.plot(x, posterior_c2_sample, 'k--', label='Class $c_2$')
plt.axvline(decision_boundary_sample, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
# plt.scatter(samples_c1, np.zeros_like(samples_c1), c='blue', marker='o', label='Samples Class 1')
# plt.scatter(samples_c2, np.zeros_like(samples_c2), c='green', marker='s', label='Samples Class 2')
plt.title('Posterior Probability')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('Posterior Probability', fontsize=14)
plt.legend()

plt.tight_layout()
plt.show()
