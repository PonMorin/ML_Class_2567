from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def F1_score(y, y_pred):
    y = np.array(y)
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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def init_theta(X):
    theta = np.zeros(X.shape[1])
    return theta

def logistic_model(X, theta):
    z = np.dot(X, theta.T)
    y_pred = sigmoid(z)
    return y_pred

def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

def cost_function(y, y_pred):
    m = len(y)
    cost = - (1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost 

def update_weight(n, old_weight, X, y, y_pred, lr=0.2):
    error = y_pred - y
    new_weight = old_weight - ((lr/n) * np.dot(X.T, error))
    return new_weight

def gradient_descent(n, X, y, theta, steps):
    cost_history = []
    for _ in range(steps):
        Y_pred = logistic_model(X, theta)
        cost = cost_function(y, Y_pred)
        cost_history.append(cost)
        theta = update_weight(n, theta, X, y, Y_pred)
    return cost_history, theta

if __name__ == "__main__":
    steps = 1000

    # Load the MNIST dataset
    mnist = fetch_openml('mnist_784')
    X, y = mnist.data, mnist.target.astype(int)

    # Filter the dataset to keep only digits 0 and 1
    X = X[np.logical_or(y == 0, y == 1)]
    y = y[np.logical_or(y == 0, y == 1)]

    # Visualize some samples before training
    y_plot = np.array(y)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X.iloc[i].values.reshape(28, 28), cmap=plt.cm.gray)
        ax.set_title(f'Label: {y_plot[i]}')
        ax.axis('off')
    plt.suptitle('Samples from MNIST Dataset Before Training')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    X_train_normal = X_train / 255.0
    X_b_train = np.c_[np.ones((len(X_train_normal), 1)), X_train_normal]

    X_test_normal = X_test / 255.0
    X_b_test = np.c_[np.ones((len(X_test_normal), 1)), X_test_normal]

    theta = init_theta(X_b_train)
    cost_history, theta = gradient_descent(len(y_train), X_b_train, y_train, theta, steps)

    y_test_pred = logistic_model(X_b_test, theta)
    y_test_pred_class = np.where(y_test_pred >= 0.5, 1, 0)
    print(y_test_pred_class)

    f1 = F1_score(y_test, y_test_pred_class)
    print(f"F1 Score: {f1}")

    theta_image = theta[1:].reshape(28, 28)
    scale = np.abs(theta_image).max()

    # Plot the coefficients
    plt.imshow(theta_image, cmap=plt.cm.bwr, vmin=-scale, vmax=scale, interpolation='bilinear')
    # plt.colorbar()
    plt.title('Coefficient of Digits 0 and 1')
    plt.show()
