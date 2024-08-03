import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def init_theta(shape):
    theta = np.array(np.zeros(shape))
    return theta

def standardization(X):
    mean_x = np.array([np.mean(X)])
    std_x = np.array([np.std(X)])
    X_sd = (X - mean_x) / std_x
    return X_sd

def linear_model(X, theta):
    return np.dot(X, theta)


def RSS_function(Y, Y_pred, ridge_lamda, theta):
    error = Y_pred - Y
    rss =  np.dot(error.T, error) + ((ridge_lamda) * np.dot(theta[-1].T, theta[-1]))
    return rss

def vitualize_data(cost, slope, ridge_lamda):
    plt.plot(slope, cost, label=f'λ = {ridge_lamda}')

if __name__ == '__main__':
    # data = pd.read_csv('./Regularization/HeightWeight100.csv')
    # X = np.array(data.iloc[:, 0].values)
    # X = standardization(X)
    # X_b = np.c_[np.ones((len(X), 1)), X]
    # Y = np.array(data.iloc[:, -1].values)

    X = np.array([0, 2])
    Y = np.array([0, 2])
    X_b = np.c_[np.ones((len(X), 1)), X]
    
    theta = np.linspace(-20, 20, 1000)
    theta_stack = np.c_[np.zeros((len(theta), 1)), theta.reshape(-1, 1)]

    list_ridgeLambda = [0, 10, 20, 40, 400]
    for ridge_lambda in list_ridgeLambda:
        rss_List = []
        for i in range(len(theta_stack)):
            y_pred = linear_model(X_b, theta_stack[i])
            rss = RSS_function(Y, y_pred, ridge_lambda, theta_stack[i])
            rss_List.append(rss)
        vitualize_data(rss_List, theta_stack[:, -1], ridge_lambda)

    plt.axis([-6, 8, -0.5, 40])
    plt.show()