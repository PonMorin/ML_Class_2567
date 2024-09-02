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
def visualize_linear_regression(X, y, w, b, w_history, b_history):
    plt.figure(figsize=(12, 5))
    
    # Linear Regression
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue')
    line_x = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    line_y = w * line_x + b
    plt.plot(line_x, line_y, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Representation')

    # สร้าง Contour Plot
    w_range = np.linspace(np.min(w_history) - 1, np.max(w_history) + 1, 100)
    b_range = np.linspace(np.min(b_history) - 1, np.max(b_history) + 1, 100)
    W, B = np.meshgrid(w_range, b_range)
    cost_values = np.array([cost_function(X, y, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
    cost_values = cost_values.reshape(W.shape)

    plt.subplot(1, 2, 2)
    plt.contour(W, B, cost_values, levels=np.logspace(-2, 3, 20))
    plt.plot(w_history, b_history, 'r-x')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title('Cost Function Contour Plot')
    plt.tight_layout()
    plt.show()

# ฟังก์ชันแสดงกราฟ Gradient Descent หลาย ๆ ค่าของ learning rate
def visualize_multiple_learning_rates(X, y, w_initial, b_initial, learning_rates, iterations):
    plt.figure(figsize=(15, 5))

    for i, lr in enumerate(learning_rates):
        w, b, cost_history, w_history, b_history = gradient_descent(X, y, w_initial, b_initial, lr, iterations)
        
        w_range = np.linspace(np.min(w_history) - 1, np.max(w_history) + 1, 100)
        cost_values = [cost_function(X, y, w, b) for w in w_range]

        plt.subplot(1, len(learning_rates), i+1)
        plt.plot(w_range, cost_values, label="MSE")
        plt.plot(w_history, cost_history, "o-", color="red", label="theta_val")  # เชื่อมต่อจุดสีแดงด้วยเส้น
        plt.xlabel("w")
        plt.ylabel("MSE")
        plt.title(f'Learning rate = {lr}')
        plt.legend()

    plt.tight_layout()
    plt.show()

# กำหนดค่าเริ่มต้นสำหรับ w และ b
w_initial = -1
b_initial = -1

# Learning rates ที่เราจะทดสอบ
learning_rates = [0.005, 0.05, 0.1, 0.2]

# กำหนดจำนวน iterations
iterations = 50

# ใช้ฟังก์ชันเพื่อแสดงผลลัพธ์
visualize_multiple_learning_rates(X, y, w_initial, b_initial, learning_rates, iterations)

# ใช้ Gradient Descent และแสดงผล Linear Regression และ Contour Plot
w_final, b_final, cost_history, w_history, b_history = gradient_descent(X, y, w_initial, b_initial, learning_rates, iterations)
visualize_linear_regression(X, y, w_final, b_final, w_history, b_history)
