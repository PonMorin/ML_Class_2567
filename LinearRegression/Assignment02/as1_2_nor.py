import numpy as np
import matplotlib.pyplot as plt

# สร้างข้อมูลตัวอย่างเริ่มต้น
X = np.array([0, 2, 5, 7])
y = np.array([0, 2, 5, 7])

# function หา Linear Regression y = wX + b
def cost_function(X, y, w, b):
    m = len(y)
    predictions = w * X + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# กำหนดค่าพารามิเตอร์และการเรียกใช้งาน Gradient Descent
w_initial = -1
b_initial = -1
learning_rate = 0.05
iterations = 100

# function ทำ Gradient Descent เพื่อปรับปรุงค่าพารามิเตอร์ w และ b ใน Linear Regression
def gradient_descent(X, y, w, b, learning_rate, iterations):
    m = len(y)
    cost_history = []  # list เก็บค่า cost ในแต่ละรอบ
    w_history = []  # list เก็บค่า w ในแต่ละรอบ
    b_history = []  # list เก็บค่า b ในแต่ละรอบ
    for i in range(iterations):  # วนลูปตามจำนวนรอบ iterations
        predictions = w * X + b  # สูตรคำนวณ
        gw = (1/m) * np.sum((predictions - y) * X)  # สูตรคำนวณ Gradient ของ w
        gb = (1/m) * np.sum(predictions - y)  # สูตรคำนวณ Gradient ของ b
        # ปรับปรุงพารามิเตอร์ w และ b โดยลบด้วยผลคูณของ learning rate กับ gw และ gb
        w -= learning_rate * gw  # ปรับปรุงค่า w โดยลบด้วยผลคูณของ learning rate กับ gw
        b -= learning_rate * gb  # ปรับปรุงค่า b โดยลบด้วยผลคูณของ learning rate กับ gb
        # เก็บค่า Cost Function ปัจจุบันที่ได้หลังจากปรับปรุงพารามิเตอร์ใน cost_history
        cost_history.append(cost_function(X, y, w, b))
        w_history.append(w)
        b_history.append(b)
    # เก็บค่าพารามิเตอร์ w และ b ใน w_history และ b_history
    return w, b, cost_history, w_history, b_history

# สร้างฟังก์ชันสำหรับแสดงผล Linear Regression และ Contour Plot
def visualize_linear_regression(X, y, w, b, w_history, b_history, title_suffix="", subplot_index=None):
    if subplot_index:
        plt.subplot(subplot_index)
    
    # Linear Regression
    plt.scatter(X, y, color='blue')
    line_x = np.linspace(np.min(X), np.max(X), 100)
    line_y = w * line_x + b
    plt.plot(line_x, line_y, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Linear Regression Representation {title_suffix}')
    plt.grid(True)

# สร้างฟังก์ชันสำหรับแสดงผล Contour Plot
def visualize_contour_plot(X, y, w_history, b_history, title_suffix="", subplot_index=None):
    if subplot_index:
        plt.subplot(subplot_index)
    
    w_range = np.linspace(np.min(w_history) - 1, np.max(w_history) + 1, 100)
    b_range = np.linspace(np.min(b_history) - 1, np.max(b_history) + 1, 100)
    W, B = np.meshgrid(w_range, b_range)
    cost_values = np.array([cost_function(X, y, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
    cost_values = cost_values.reshape(W.shape)

    plt.contour(W, B, cost_values, levels=np.logspace(-2, 3, 20))
    plt.plot(w_history, b_history, 'r-x')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title(f'Cost Function Contour Plot {title_suffix}')
    plt.grid(True)

# Gradient Descent on original data
w_final, b_final, cost_history, w_history, b_history = gradient_descent(X, y, w_initial, b_initial, learning_rate, iterations)

# Standardize data
X_standardized = (X - np.mean(X)) / np.std(X)

# Gradient Descent on standardized data
w_final_std, b_final_std, cost_history_std, w_history_std, b_history_std = gradient_descent(X_standardized, y, w_initial, b_initial, learning_rate, iterations)

# แสดงผลในรูปแบบ 2x2
plt.figure(figsize=(12, 10))

# Original data
visualize_linear_regression(X, y, w_final, b_final, w_history, b_history, title_suffix="(Original)", subplot_index=221)
visualize_contour_plot(X, y, w_history, b_history, title_suffix="(Original)", subplot_index=222)

# Standardized data
visualize_linear_regression(X_standardized, y, w_final_std, b_final_std, w_history_std, b_history_std, title_suffix="(Standardized)", subplot_index=223)
visualize_contour_plot(X_standardized, y, w_history_std, b_history_std, title_suffix="(Standardized)", subplot_index=224)

plt.tight_layout()
plt.show()
