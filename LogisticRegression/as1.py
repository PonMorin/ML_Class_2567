import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
    return theta 


def predict(X, theta):
    z = np.dot(X, theta)
    h = sigmoid(z)
    return np.round(h)

# สร้างข้อมูลสำหรับปัญหา AND, OR, XOR
def create_data(problem):
    if problem == "AND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
    elif problem == "OR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
    elif problem == "XOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        # เพิ่ม interaction feature สำหรับ XOR
        X = np.c_[X, X[:, 0] * X[:, 1]]
    return X, y

# เลือกปัญหา
problem = "XOR"  # เปลี่ยนเป็น "AND" หรือ "OR" ได้

# สร้างข้อมูลและฝึกโมเดล
X, y = create_data(problem)
theta = gradient_descent(X, y, learning_rate=0.1, num_iterations=1000)

# ทำนายผลลัพธ์
y_pred = predict(X, theta)

print("X:", X)
print("y:", y)
print("y_pred:", y_pred)