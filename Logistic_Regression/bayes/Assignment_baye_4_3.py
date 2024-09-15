import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# กำหนดค่าพารามิเตอร์ของการแจกแจง
mu_c1 = -1
mu_c2 = 1
sigma = 1

# สร้างฟังก์ชัน Linear Discriminant
def linear_discriminant(mu_c1, mu_c2, sigma):
    w = (mu_c2 - mu_c1) / sigma**2
    b = -0.5 * (mu_c1**2 - mu_c2**2) / sigma**2
    return w, b

# คำนวณ w และ b
w, b = linear_discriminant(mu_c1, mu_c2, sigma)

# สร้างกราฟ
x = np.linspace(-4, 4, 400)

# คำนวณ likelihood
likelihood_c1 = norm.pdf(x, mu_c1, sigma)
likelihood_c2 = norm.pdf(x, mu_c2, sigma)

# คำนวณ posterior โดยสมมติว่า p(c1) = p(c2) = 0.5
posterior_c1 = likelihood_c1 / (likelihood_c1 + likelihood_c2)
posterior_c2 = likelihood_c2 / (likelihood_c1 + likelihood_c2)

# คำนวณขอบตัดสินใจ
decision_boundary_sample = -b / w

# วาดกราฟ likelihood
plt.figure(figsize=(18, 6))

plt.subplot(2, 1, 1)
plt.plot(x, likelihood_c1, 'k-', label='Class $c_1$')
plt.plot(x, likelihood_c2, 'k--', label='Class $c_2$')
plt.axvline(decision_boundary_sample, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
plt.title('Likelihood')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('Likelihood', fontsize=14)
plt.legend()

# วาดกราฟ posterior
plt.subplot(2, 1, 2)
plt.plot(x, posterior_c1, 'k-', label='Class $c_1$')
plt.plot(x, posterior_c2, 'k--', label='Class $c_2$')
plt.axvline(decision_boundary_sample, color='black', linestyle='-', linewidth=2, label='Decision Boundary')
plt.title('Posterior Probability')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('Posterior Probability', fontsize=14)
plt.legend()

plt.tight_layout()
plt.show()