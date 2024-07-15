import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def problem_data(X):
    return np.sin(np.pi * X)

def init_theta(shape):
    theta = np.zeros(shape)
    return theta

def cost_function(n, Y, Y_pred):
    cost = (1 / (2 * n)) * np.sum((Y_pred - Y)**2)
    return cost

def linear_model(X, theta):
    return np.dot(X, theta)

def normal_equation(X, Y):
    theta = np.dot(pinv(np.dot(X.T, X)),  np.dot(X.T, Y))
    return theta

def cal_mean_model(Ed_model):
    mean_model = np.mean(Ed_model, axis=0)
    return mean_model

if __name__ == "__main__":
    X = np.linspace(-1, 1)  # Use 100 points between -1 and 1

    X_b = np.c_[np.ones((len(X), 1)), X]
    
    y = problem_data(X)
    steps = 10000

    training_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    list_Ein = []
    list_Eout = []

    for m in training_sizes:
        Ein_steps = []
        ed_prediction_list = []
        
        for _ in range(steps):
            random_samples_X = np.random.choice(X, m, replace=False)
            y_sample = problem_data(random_samples_X)

            X_b_sample = np.c_[np.ones((len(random_samples_X), 1)), random_samples_X]
           
            # Normal Equation
            normal_theta = normal_equation(X_b_sample, y_sample)
            theta_arr = np.array(normal_theta)

            # Find y_pred to cal Eout
            sample_linear = linear_model(X_b, theta_arr)
            ed_prediction_list.append(sample_linear)

            # Find Cost for Ein
            train_linear = linear_model(X_b_sample, theta_arr)
            e = cost_function(m, y_sample, train_linear)
            Ein_steps.append(e)
            

        ed_arr_prediction = np.array(ed_prediction_list)
        mean_model = cal_mean_model(ed_arr_prediction)
        bias_square = np.mean(np.square(mean_model - y))
        var_x = np.mean(np.square(ed_arr_prediction - mean_model))
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
