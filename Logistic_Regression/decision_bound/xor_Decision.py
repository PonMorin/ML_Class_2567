# Pon
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def standardization(X):
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    X_sd = (X - mean_x) / std_x
    return X_sd

def F1_score(y, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f1_score

def init_theta(X_shape):
    theta = np.zeros(X_shape)
    return theta

def logistic_model(X, theta):
    z = np.dot(X, theta)
    y_pred = sigmoid(z)
    return y_pred

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(y, y_pred):
    cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost 

def update_weight(n, old_weight, X, y, y_pred, lr=0.3):
    error = y_pred - y
    new_weight = old_weight - ((lr / n) * (np.dot(X.T, error)))
    return new_weight

def gradient_descent(n, X, y, theta, steps):
    cost_history = []
    theta_list = []
    for _ in range(steps):
        Y_pred = logistic_model(X, theta)
        cost = cost_function(y, Y_pred)
        cost_history.append(cost)
        theta = update_weight(n, theta, X, y, Y_pred)
        theta_list.append(theta)
    return cost_history, theta_list

if __name__ == "__main__":
    steps = 1000

    # XOR Gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0]) 

    # Interaction Feature
    X_interaction = np.c_[X, X[:, 0] * X[:, 1]]
    X_b = np.c_[np.ones((len(X_interaction), 1)), X_interaction]
    n = len(X)

    theta = init_theta(X_b.shape[1])

    cost_history, theta_list = gradient_descent(n, X_b, y, theta, steps)
    theta = theta_list[-1]

    Y_pred = logistic_model(X_b, theta)

    Y_pred_labels = (Y_pred > 0.5).astype(int)

    print("Predicted Labels:", Y_pred_labels)

    dataset_array = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    dataset_df = pd.DataFrame(dataset_array, columns=['Col 1', 'Col 2', 'Target'])

    # Create Decision Boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = logistic_model(np.c_[np.ones((xx.ravel().shape[0], 1)),
                             xx.ravel(), yy.ravel(), (xx * yy).ravel()], theta)
    Z = Z.reshape(xx.shape)

    # Plotting the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='k', cmap=plt.cm.Spectral)
    plt.title("Decision Boundary XOR")
    plt.xlabel("Col 1")
    plt.ylabel("Col 2")
    plt.show()