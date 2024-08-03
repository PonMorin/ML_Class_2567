import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def problem_data(X):
    noise = np.random.normal(0, 0.3, X.shape)
    return np.sin(np.pi * X) + noise

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
    steps = 1000

    training_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    list_Ein = []
    list_Eout = []
    bias_list = []

    for m in training_sizes:
        Ein_steps = []
        Eout_steps = []
        ed_prediction_list = []
        
        for _ in range(steps):
            random_samples_X = np.random.choice(X, m, replace=False)
            y_sample = problem_data(random_samples_X)

            X_b_sample = np.c_[np.ones((len(random_samples_X), 1)), random_samples_X]
           
            # Normal Equation
            normal_theta = normal_equation(X_b_sample, y_sample)
            theta_arr = np.array(normal_theta)

            # Find Cost for Ein
            train_linear = linear_model(X_b_sample, theta_arr)
            e = cost_function(m, y_sample, train_linear)
            Ein_steps.append(e)
            
            # Find y_pred to cal Eout
            normal_theta_val = normal_equation(X_b, y)
            theta_arr_val = np.array(normal_theta_val)
            sample_linear = linear_model(X_b, theta_arr_val)
            ed_prediction_list.append(sample_linear)
            val_e = cost_function(m, y, sample_linear)
            Eout_steps.append(val_e)

        ed_arr_prediction = np.array(ed_prediction_list)
        mean_model = cal_mean_model(ed_arr_prediction)

        list_Ein.append(np.mean(Ein_steps))
        list_Eout.append(np.mean(Eout_steps))

        bias = np.mean(np.square(mean_model - y))
        bias_list.append(bias)

    bias_mean = np.mean(np.array(bias_list))
    plt.figure()
    plt.plot(X, y, c="#4CAF50")
    plt.figure()
    plt.plot(training_sizes, np.array(bias_list), label=f'Bias: {bias_mean:.2f}', linestyle='--')
    plt.plot(training_sizes, np.array(list_Ein), label='Training Error (Ein)', c='r')
    plt.plot(training_sizes, np.array(list_Eout), label='Validation Error (Eout)', c='b')
    plt.xlabel('Training Set Size')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Learning Curve')
    plt.show()