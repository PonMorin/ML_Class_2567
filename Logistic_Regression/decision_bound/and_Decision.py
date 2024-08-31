# Pon
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def standardization(X):
    mean_x = np.array([np.mean(X)])
    std_x = np.array([np.std(X)])
    X_sd = (X - mean_x) / std_x
    return X_sd

def F1_score(y, y_pred):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y[i] == 0 and y_pred[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return f1_score

def init_theta(X):
    theta = np.array(np.zeros(X))
    return theta

def logistic_model(X, theta):
    z = np.dot(X, theta)
    y_pred = sigmoid(z)
    return y_pred

def sigmoid(z):
    sig = 1 / (1 + np.e**(-z))
    return sig

def cost_function(y, y_pred):
    cost = np.dot(-y.T, np.log(y_pred)) - np.dot((1-y).T, np.log(1 - y_pred))
    return cost 

def update_weight(n, old_weight, X, y, y_pred, lr=0.3):
    error = y_pred - y
    new_weight = old_weight - ( (lr/n) * (np.dot(X.T, error)) )
    return new_weight

def gradient_descent(n, X, y, theta, steps):
    cost_history = []
    Y_pred = logistic_model(X, theta)
    theta_list = []
    cost_history.append(1e10)

    for _ in range(1, steps+1):
        Y_pred = logistic_model(X, theta)

        cost = cost_function(y, Y_pred)
        cost_history.append(cost)
        
        theta = update_weight(n, theta, X, y, Y_pred)
        theta_list.append(theta)

    cost_history.pop(0)            
        
    return cost_history, theta_list

if __name__ == "__main__":
    steps = 100

    # AND Gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    X_b = np.c_[np.ones((len(X), 1)), X]

    n = len(X)

    theta = init_theta(X_b.shape[1])

    cost_history, theta_list = gradient_descent(n, X_b, y, theta, steps)
    theta = theta_list[-1]

    Y_pred = logistic_model(X_b, theta)

    print(Y_pred)

    for i in range(len(Y_pred)):
        if Y_pred[i] > 0.5:
            Y_pred[i] = 1
        else:
            Y_pred[i] = 0
    print(Y_pred)
    # print(theta)

    dataset_array = np.concatenate((X, y.reshape(-1,1)), axis=1)
    dataset_df = pd.DataFrame(dataset_array, columns = ['Col 1', 'Col 2', 'Target'])

    # Create Decision Boundary
    x1_max, x1_min = X[:, 0].max(), X[:, 0].min()
    x2_max, x2_min = X[:, 1].max(), X[:, 1].min()
    x_vals = np.array([-4, 5])
    slope = - theta[1] / theta[2]
    intercept = - theta[0] / theta[2]
    decision_boundary = slope * x_vals + intercept

    # Plot the dataset with decision boundary
    plt.figure(figsize=(12,8))
    sns.scatterplot(data=dataset_df, x='Col 1', y='Col 2', hue='Target')
    plt.plot(x_vals, decision_boundary, linestyle='--', color='black', label='Decision Boundary')
    plt.fill_between(x_vals, decision_boundary, x2_min-5, color='tab:blue', alpha=0.2)
    plt.fill_between(x_vals, decision_boundary, x2_max+5, color='tab:orange', alpha=0.2)
    plt.xlabel("Column 1")
    plt.ylabel("Column 2")
    plt.legend(loc='best')
    plt.show()
