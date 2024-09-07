# กำหนดค่า
import numpy as np
import matplotlib.pyplot as plt

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
decision_boundary = w[0] * X1 + w[1] * X2 + b

plt.figure(figsize=(8, 8))
plt.contour(X1, X2, decision_boundary, levels=[0], colors='black')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.title('ขอบตัดสินใจโดยใช้ค่าพารามิเตอร์ที่กำหนด', fontsize=16, fontname='Tahoma')
plt.grid(True)
plt.show()