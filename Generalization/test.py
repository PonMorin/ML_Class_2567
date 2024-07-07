import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def problem_data(X):
    return np.sin(np.dot(np.pi, X))

def init_theta(shape):
    theta = np.array(np.zeros(shape))
    return theta

def cost_function(n, Y, Y_pred):
    error = Y_pred - Y
    cost = (1/2*n) * np.dot(error.T, error)
    return cost

def linear_model(X, theta):
    return np.dot(X, theta)

def normal_equation(X, Y):
    theta = np.dot(inv(np.dot(X.T, X)),  np.dot(X.T, Y))
    return theta

if __name__ == "__main__":
    X = np.linspace(-1, 1)
    y = problem_data(X)

    steps = 100000
    model_list = list()
    for i in range(steps):
        random_samples_X = np.random.choice(X, 2, replace=False)
        y_sample = problem_data(random_samples_X)

        X_b = np.c_[np.ones((len(random_samples_X), 1)), random_samples_X]
        n = len(random_samples_X)
        theta = init_theta(X_b.shape[1])

        # Normal Equation
        normal_theta = normal_equation(X_b, y_sample)
        normal_y_pred = linear_model(X_b, normal_theta)
        normal_cost = cost_function(n, y_sample, normal_y_pred)
        
        model_list.append(normal_theta)
    
    # print(len(model_list))
    theta0, theta1 = 0, 0
    for i in range(len(model_list)):
        theta0 += model_list[i][0]
        theta1 += model_list[i][1]

    mean_theta0 = np.mean(theta0)
    mean_theta1 = np.mean(theta1)
    mean_model = np.array([mean_theta0, mean_theta1])

    var_X = np.square(model_list) - np.square(mean_model)
    print(var_X)
    bias_2 = np.square(mean_model - y)
    print(bias_2)
    # print(test_minus)
        

    
    # plt.plot(X, y, c="#4CAF50")
    # plt.scatter(random_samples_X, y_sample, c="red")
    # plt.plot(random_samples_X, normal_y_pred, c="black")
    # plt.show()