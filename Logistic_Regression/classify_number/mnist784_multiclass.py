from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def standardization(X):
    mean_x = np.mean(X)
    std_x = np.std(X)
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
    return 1 / (1 + np.exp(-z))

def cost_function(y, y_pred):
    m = len(y)
    cost = - (1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost 

def update_weight(n, old_weight, X, y, y_pred, lr=0.2):
    error = y_pred - y
    new_weight = old_weight - ( (lr/n) * np.dot(X.T, error))
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

    mnist = fetch_openml('mnist_784')
    X, y = mnist.data, mnist.target.astype(int)
    
    # # Visualize some samples before training
    # fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    # for i, ax in enumerate(axes.ravel()):
    #     ax.imshow(X.iloc[i].values.reshape(28, 28), cmap=plt.cm.gray)
    #     ax.set_title(f'Label: {y[i]}')
    #     ax.axis('off')
    # plt.suptitle('Samples from MNIST Dataset Before Training')
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


    X_train_normal = X_train / 255.0
    X_b_train = np.c_[np.ones((len(X_train_normal), 1)), X_train_normal]

    X_test_normal = X_test / 255.0
    X_b_test = np.c_[np.ones((len(X_test_normal), 1)), X_test_normal]

    n_classes = len(np.unique(y_train))
    all_theta = np.zeros((n_classes, X_b_train.shape[1]))

    for i in range(n_classes):
        y_train_class = np.where(y_train == i, 1, 0)
        theta = init_theta(X_b_train)
        cost_history, theta = gradient_descent(len(y_train_class), X_b_train, y_train_class, theta, steps)
        all_theta[i] = theta

    y_pred_test = logistic_model(X_b_test, all_theta)
    y_test_pred_class = np.where(y_pred_test >= 0.5, 1, 0)
    print(y_test_pred_class[3])

    # Plotting the coefficients
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.ravel()

    for i in range(n_classes):
        theta_image = all_theta[i, 1:].reshape(28, 28)
        scale = np.abs(theta_image).max()

        axes[i].imshow(theta_image, cmap=plt.cm.bwr, vmin=-scale, vmax=scale, interpolation='bilinear')
        axes[i].set_title(f'Class {i}')
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle('Coefficient of Digits 0 - 9')
    plt.show()
