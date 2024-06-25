# Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt

def init_theta(shape):
    theta = np.array(np.zeros(shape))
    return theta

def linear_model(X, theta):
    return np.dot(X, theta)

def cost_function(n, Y, Y_pred):
    error = Y_pred - Y
    cost = (1/2*n) * np.dot(error.T, error)
    return cost

def update_weight(n, old_weight, X, y_pred, Y, lr=0.1):
    error = y_pred - Y
    new_weight = old_weight - ( (lr/n) * (np.dot(X.T, error)))
    return new_weight

def gradient_descent(n, X, Y, theta, epochs):

    cost_history = []
    Y_pred = linear_model(X, theta)
    theta_list = []
    cost_history.append(1e10)

    for _ in range(1, epochs+1):
        Y_pred = linear_model(X, theta)

        cost = cost_function(n, Y_pred, Y)
        cost_history.append(cost)
        
        theta = update_weight(n, theta, X, Y_pred, Y)
        theta_list.append(theta)

    cost_history.pop(0)            
        
    return cost_history, theta_list

def vitualize_data(X, Y, y_pred, cost, epochs):
    theta0_range = np.linspace(-10, 10, 100)
    theta1_range = np.linspace(-10, 10, 100)

    Fake_theta0, Fake_theta1 = np.meshgrid(theta0_range, theta1_range)
    cost = np.zeros(Fake_theta0.shape)


    for i in range(len(theta0_range)):
        for j in range(len(theta1_range)):
            y_pred = np.dot(X, Fake_theta1[i, j]) + Fake_theta0[i, j]
            error = y_pred - Y
            cost[i, j] = (1/(2*n)) * np.sum(np.square(error))

    theta0 = []
    theta1 = []
    for i in range(100):
        theta0.append(theta_list[i][0])
        theta1.append(theta_list[i][1])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6.15))

    ax[0, 0].plot(X, Y_pred)
    ax[0, 0].scatter(X, Y)
    ax[0, 0].set_xlabel('X')
    ax[0, 0].set_ylabel('Y')

    ax[0, 1].contour(Fake_theta0, Fake_theta1, cost, cmap='rainbow')
    ax[0, 1].scatter(theta0, theta1)
    ax[0, 1].set_xlabel('theta0')
    ax[0, 1].set_ylabel('theta1')

    ax[1, 0].plot(range(1, epochs + 1), cost_history, color='blue')
    ax[1, 0].set_xlabel('epochs')
    ax[1, 0].set_ylabel('J')

    plt.show()

if __name__ == '__main__':
    epochs = 100    # Step for train model

    X = np.array([0, 2])
    X_b = np.c_[np.ones((len(X), 1)), X]

    Y = np.array([0, 2])

    n = len(X)

    theta = init_theta(X_b.shape[1])    #initialize Theta
    
    cost_history, theta_list = gradient_descent(n, X_b, Y, theta, epochs)
    theta = theta_list[-1]
    Y_pred = linear_model(X_b, theta)
    vitualize_data(X, Y, Y_pred, cost_history, epochs)
    