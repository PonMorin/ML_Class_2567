import numpy as np
import matplotlib.pyplot as plt

def init_weight(shape):
    theta = np.array(np.zeros(shape))
    return theta

def standardization(X):
    mean_x = np.array([np.mean(X)])
    std_x = np.array([np.std(X)])
    X_sd = (X - mean_x) / std_x
    return X_sd

def linear_model(X, theta):
    xw = np.dot(X, theta)
    return xw

def cost_function(n, Y_pred, Y):
    error = Y_pred - Y
    cost = (1/2*n) * np.dot(error.T, error)
    return cost

def update_weight(n, old_weight, X, y_pred, Y, lr=0.1):
    error = y_pred - Y
    new_weight = old_weight - ( (lr/n) * (np.dot(X.T, error)))
    return new_weight

def gradient_descent(n, X, Y, theta, epochs):
    cost_list = []
    theta_list = []
    prediction_list = []

    cost_list.append(1e10) # init cost
    for _ in range(epochs):
        y_pred = linear_model(X, theta)
        prediction_list.append(y_pred)

        cost = cost_function(n, y_pred, Y)
        cost_list.append(cost)

        theta = update_weight(n, theta, X, y_pred, Y)
        theta_list.append(theta)

    cost_list.pop(0)
    return prediction_list, cost_list, theta_list

def contour_plot(n, X, Y, real_Theta, axis):
    theta0_range = np.linspace(-100, 100, 100)
    theta1_range = np.linspace(-100, 100, 100)

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
        theta0.append(real_Theta[i][0])
        theta1.append(real_Theta[i][1])
    axis.contour(Fake_theta0, Fake_theta1, cost, cmap='rainbow')
    axis.scatter(theta0, theta1)
    axis.set_title('Contour')
    axis.set_xlabel("theta0")
    axis.set_ylabel("theta1")

def vitualize_data(n, X, Y, theta_list):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6.15))

    ax[0].plot(range(1, epochs+1), cost_list)
    ax[0].set_title('Epochs Vs Cost')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('J')

    contour_plot(n, X, Y, theta_list, ax[1])

    plt.show()

if __name__ == "__main__":    
    # X = np.array([[2104, 1416, 1534, 852], 
    #               [5, 3, 3, 2], 
    #               [1, 2, 2, 1], 
    #               [45, 40, 30, 36]])

        # Y = np.array([460, 
    #               232, 
    #               315, 
    #               178])

    epochs = 100

    X = np.array([0, 200])
    X = standardization(X)
    X_b = np.c_[np.ones((len(X), 1)), X]

    Y = np.array([0, 2])
    n = len(Y)

    theta = init_weight(X_b.shape[1])
    
    prediction_list, cost_list, theta_list = gradient_descent(n, X_b, Y, theta, epochs)
    theta = theta_list[-1]  # Get last theta value

    vitualize_data(n, X, Y, theta_list)    # Display Graph

    
    