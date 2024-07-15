import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy import integrate

def problem_data(X):
    noise = np.random.normal(0, 0.3, X.shape)
    return np.sin(np.dot(np.pi, X)) + noise

def constant_model(X):
    return np.mean(np.sin(np.dot(np.pi, X)))

def cal_mean_model(period):
    # result, _= integrate.dblquad(constant_model, -1, 1, lambda x: -1, lambda x: 1)
    # mean_model = result / period
    mean_model = 0
    return mean_model

def cost_function(n, Y, Y_pred):
    cost = (1 / (2 * n)) * np.sum((Y_pred - Y)**2)
    return cost

if __name__ == "__main__":
    X = np.linspace(-1, 1)
    y = problem_data(X)

    steps = 10000

    training_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    list_Ein = []
    list_Eout = []

    for m in training_sizes:
        Ein_steps = []
        prediction_list = []
        
        for _ in range(steps):
            random_samples_X = np.random.choice(X, m, replace=False)
            y_sample = problem_data(random_samples_X)

            sample_constant = constant_model(random_samples_X)
            prediction_list.append(sample_constant)

            # Find Cost for Ein
            train_constant = constant_model(random_samples_X)
            e = cost_function(m, y_sample, train_constant)
            Ein_steps.append(e)
            

        prediction_arr = np.array(prediction_list)
        mean_model = cal_mean_model(steps)

        bias_square = np.mean(np.square(mean_model - y))
        var_x = np.mean(np.square(prediction_arr - mean_model))
        variance = np.mean(var_x)
        E_out = bias_square + variance
        
        list_Ein.append(np.mean(np.array([Ein_steps])))
        list_Eout.append(E_out)
    print(E_out)
    plt.figure()
    plt.plot(X, y, c="#4CAF50")
    plt.figure()
    plt.plot(training_sizes, np.array(list_Ein), label='Training Error (Ein)', c='r')
    plt.plot(training_sizes, np.array(list_Eout), label='Validation Error (Eout)', c='b')
    plt.xlabel('Training Set Size')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()