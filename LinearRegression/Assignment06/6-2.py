import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hypothesis function for logistic regression
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# Loss function (Binary Cross-Entropy)
def compute_loss(y, y_pred):
    m = len(y)
    loss = -(1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

# Gradient Descent for Logistic Regression
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    loss_history = []

    for i in range(iterations):
        y_pred = hypothesis(X, theta)
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)

        # Gradient calculation
        gradient = (1/m) * np.dot(X.T, (y_pred - y))
        
        # Update parameters
        theta -= alpha * gradient

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
    
    return theta, loss_history

# Main function to perform logistic regression
def logistic_regression(X, y, alpha, iterations):
    # Initialize parameters
    X = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for the intercept term
    theta = np.zeros(X.shape[1])
    
    # Perform gradient descent
    theta, loss_history = gradient_descent(X, y, theta, alpha, iterations)
    
    return theta, loss_history

# Predict function
def predict(X, theta):
    X = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for the intercept term
    y_pred = hypothesis(X, theta)
    return (y_pred >= 0.5).astype(int)

# Function to plot decision boundary
def plot_decision_boundary(X, y, theta, title):
    plt.figure(figsize=(6, 6))
    
    # Plot data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

    # Create grid to evaluate model
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

    X_grid = np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()]
    y_grid_pred = hypothesis(X_grid, theta).reshape(xx1.shape)
    
    # Plot decision boundary
    plt.contour(xx1, xx2, y_grid_pred, levels=[0.5], cmap="Greys", vmin=0, vmax=1)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Function to generate synthetic datasets
def generate_datasets():
    # Generating a linearly separable dataset
    X1, y1 = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # Generating a non-linearly separable dataset
    X2, y2 = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=0.5, random_state=42)

    return (X1, y1), (X2, y2)

# Train and evaluate logistic regression models on generated datasets
def train_and_evaluate_generated_data():
    (X1, y1), (X2, y2) = generate_datasets()

    # Hyperparameters
    alpha = 0.1  # Learning rate
    iterations = 1000  # Number of iterations

    print("Training model on linearly separable dataset...")
    theta1, _ = logistic_regression(X1, y1, alpha, iterations)
    print(f"Trained parameters (theta) for linearly separable dataset: {theta1}")
    predictions1 = predict(X1, theta1)
    print(f"Predictions: {predictions1}")
    print(f"Actual labels: {y1}\n")
    plot_decision_boundary(X1, y1, theta1, "Decision Boundary for Linearly Separable Data")

    print("Training model on non-linearly separable dataset...")
    theta2, _ = logistic_regression(X2, y2, alpha, iterations)
    print(f"Trained parameters (theta) for non-linearly separable dataset: {theta2}")
    predictions2 = predict(X2, theta2)
    print(f"Predictions: {predictions2}")
    print(f"Actual labels: {y2}\n")
    plot_decision_boundary(X2, y2, theta2, "Decision Boundary for Non-Linearly Separable Data")

# Run training and evaluation on generated data
if __name__ == "__main__":
    train_and_evaluate_generated_data()
