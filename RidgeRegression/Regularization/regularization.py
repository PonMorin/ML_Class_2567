import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def problem_data(X):
    return np.sin(np.dot(np.pi, X))

def init_theta(shape):
    theta = np.array(np.zeros(shape))
    return theta

def linear_model(X, theta):
    return np.dot(X, theta)

def normal_equation(X, Y):
    theta = np.dot(pinv(np.dot(X.T, X)),  np.dot(X.T, Y))
    return theta

def cal_mean_model(Ed_model):
    # mean_theta = np.array([np.mean(all_theta[:, 0]), np.mean(all_theta[:, 1])])
    mean_model = np.mean(Ed_model, axis=0)
    return mean_model

def unRegularization(X, y, ax):
    X_b = np.c_[np.ones((len(X), 1)), X]

    steps = 1000

    ed_prediction_list = list()
    
    for _ in range(steps):
        random_samples_X = np.random.choice(X, 2)
        y_sample = problem_data(random_samples_X)

        X_b_sample = np.c_[np.ones((len(random_samples_X), 1)), random_samples_X]
        n = len(random_samples_X)

        # Normal Equation
        normal_theta = normal_equation(X_b_sample, y_sample)

        theta_arr = np.array(normal_theta)
        sample_linear = linear_model(X_b, theta_arr)
        ed_prediction_list.append(sample_linear)
        ax.plot(X, sample_linear, c="black", alpha=0.01)
    
    ed_arr_prediction = np.array(ed_prediction_list)
    mean_model = cal_mean_model(ed_arr_prediction)

    bias_square = np.mean(np.square(mean_model - y))
    var_x = np.mean(np.square(ed_arr_prediction - mean_model))
    variance = np.mean(var_x)
    E_out = bias_square + variance
    
    print("Without Regularization")
    print("Bias:", bias_square)
    print("Variance:", variance)
    print("E_out:", E_out)

    ax.plot(X, y, c="#4CAF50")
    ax.plot(X, mean_model, c='red')
    ax.set_title(f"Bias = {bias_square:.2f}, Variance = {variance:.2f}, E_out = {E_out:.2f}")
    ax.set_ylim(-2, 2)

def ridge_normal_equation(X, Y, ridge_lambda, identity):
    theta = np.dot(pinv(np.dot(X.T, X) + np.dot(ridge_lambda, identity)),  np.dot(X.T, Y))
    return theta

def regularization(X, y, ax):
    X_b = np.c_[np.ones((len(X), 1)), X]

    identity = np.identity(X_b.shape[1])
    identity[0][0] = 0
    
    steps = 1000

    ed_prediction_list = list()

    ridge_lambda = 0.5
    
    for _ in range(steps):
        random_samples_X = np.random.choice(X, 2)
        y_sample = problem_data(random_samples_X)

        X_b_sample = np.c_[np.ones((len(random_samples_X), 1)), random_samples_X]
        n = len(random_samples_X)

        # Normal Equation
        normal_theta = ridge_normal_equation(X_b_sample, y_sample, ridge_lambda, identity)

        theta_arr = np.array(normal_theta)
        sample_linear = linear_model(X_b, theta_arr)
        ed_prediction_list.append(sample_linear)
        ax.plot(X, sample_linear, c="black", alpha=0.01)
    
    ed_arr_prediction = np.array(ed_prediction_list)
    mean_model = cal_mean_model(ed_arr_prediction)

    bias_square = np.mean(np.square(mean_model - y))
    var_x = np.mean(np.square(ed_arr_prediction - mean_model))
    variance = np.mean(var_x)
    E_out = bias_square + variance
    
    print("\nWith Regularization")
    print("Bias:", bias_square)
    print("Variance:", variance)
    print("E_out:", E_out)

    ax.plot(X, y, c="#4CAF50")
    ax.plot(X, mean_model, c='red', label=f'Lambda = {ridge_lambda}')
    ax.set_title(f"Bias = {bias_square:.2f}, Variance = {variance:.2f}, E_out = {E_out:.2f}")
    ax.legend(loc="upper right")
    ax.set_ylim(-2, 2)
    

if __name__ == "__main__":
    X = np.linspace(-1, 1)
    y = problem_data(X)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    unRegularization(X, y, ax[0])
    regularization(X, y, ax[1])
    plt.show()

    