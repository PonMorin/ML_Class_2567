import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy import integrate

def problem_data(X):
    return np.sin(np.dot(np.pi, X))

def constant_model(X, X2):
    return ( np.sin(np.pi * X) + np.sin(np.pi * X2) )/ 2

def cal_mean_model(period):
    result, _= integrate.dblquad(constant_model, -1, 1, lambda x: -1, lambda x: 1)
    mean_model = result / period
    return mean_model

def cal_bias_square():
    pass

if __name__ == "__main__":
    X = np.linspace(-1, 1)
    y = problem_data(X)

    steps = 100

    prediction_list = list()
    
    for _ in range(steps):
        random_samples_X = np.random.choice(X, 2)
        y_sample = problem_data(random_samples_X)

        sample_constant = constant_model(random_samples_X[0], random_samples_X[1])
        prediction_list.append(sample_constant)
        plt.axhline(sample_constant)
    
    prediction_arr = np.array(prediction_list)
    mean_model = cal_mean_model(steps)

    bias_square = np.mean(np.square(mean_model - y))
    var_x = np.mean(np.square(prediction_arr - mean_model))
    variance = np.mean(var_x)
    E_out = bias_square + variance
    print("Bias:", bias_square)
    print("Variance:", variance)
    print("E_out:", E_out)

    plt.plot(X, y, c="#4CAF50")
    # plt.scatter(random_samples_X, y_sample)
    # plt.plot(X, mean_model, c='red')
    plt.axhline(mean_model, c='red')
    # plt.ylim(-1, 1)
    plt.show()