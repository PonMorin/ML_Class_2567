# Compare Normal Equation and Gradient Descent
import numpy as np
from numpy.linalg import inv
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

def normal_equation(X, Y):
    theta = np.dot(inv(np.dot(X.T, X)),  np.dot(X.T, Y))
    return theta

def visualize_data(X, Y, normal_y_pred, gradient_y_pred):
    # Plot for compare
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))

    ax[0].scatter(X, Y)
    ax[0].plot(X, gradient_y_pred)
    ax[0].set_title('Gradient Descent')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].scatter(X, Y)
    ax[1].plot(X, normal_y_pred)
    ax[1].set_title('Normal Equation')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')

    plt.show()

if __name__ == '__main__':

    epochs = 100

    # X = np.array([29, 28, 34, 31, 25])
    # Y = np.array([77, 62, 93, 84, 59])

    X = np.array([0, 2])
    Y = np.array([0, 2])

    X_b = np.c_[np.ones((len(X), 1)), X]
    n = len(X)
    theta = init_theta(X_b.shape[1])

    # Normal Equation
    normal_theta = normal_equation(X_b, Y)
    normal_y_pred = linear_model(X_b, normal_theta)
    normal_cost = cost_function(n, Y, normal_y_pred)

    # Gradient Descent
    cost_history, theta_list = gradient_descent(n, X_b, Y, theta, epochs)
    theta = theta_list[-1]
    Y_pred = linear_model(X_b, theta)


    visualize_data(X, Y, normal_y_pred, Y_pred) #Display Graph