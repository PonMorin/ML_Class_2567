import numpy as np
import matplotlib.pyplot as plt

µ1, σ1 = -1, 1  # ค่าเฉลี่ย(µ) and ส่วนเบี่ยงเบนมาตราฐาน(σ) ของ class c1
µ2, σ2 = 1, 1   # ค่าเฉลี่ย(µ) and ส่วนเบี่ยงเบนมาตราฐาน(σ) ของ class c2
prior_c1 = 0.5       # ความน่าจะเป็นก่อนหน้า ของ class c1
prior_c2 = 0.5       # ความน่าจะเป็นก่อนหน้า ของ class c2

x = np.linspace(-4, 4, 1000) 
# Likelihood functions
p_x_given_c1 = (1 / np.sqrt(2 * np.pi * σ1**2)) * np.exp(-((x - µ1)**2) / (2 * σ1**2))
p_x_given_c2 = (1 / np.sqrt(2 * np.pi * σ2**2)) * np.exp(-((x - µ2)**2) / (2 * σ2**2))
# Posterior probabilities using Bayes' Theorem
p_c1_given_x = (p_x_given_c1 * prior_c1) / (p_x_given_c1 * prior_c1 + p_x_given_c2 * prior_c2)
p_c2_given_x = (p_x_given_c2 * prior_c2) / (p_x_given_c1 * prior_c1 + p_x_given_c2 * prior_c2)
# Plotting the Likelihood functions
plt.figure(figsize=(10, 8))
plt.plot(x, p_x_given_c1, label="Class $c_1$")
plt.plot(x, p_x_given_c2, label="Class $c_2$", linestyle='--')
plt.title("Likelihood")
plt.xlabel("$x$")
plt.ylabel("Likelihood $p(x|c_i)$")
plt.legend()
plt.show()
plt.plot(x, p_c1_given_x, label="Class $c_1$")
plt.plot(x, p_c2_given_x, label="Class $c_2$", linestyle='--')
plt.title("Posterior Probability")
plt.xlabel("$x$")
plt.ylabel("Posterior Probability $p(c_i|x)$")
plt.legend()
plt.show()