import numpy as np
import matplotlib.pyplot as plt

# g(z) = 1/(1+(e**(-z)))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ℎ(x) = g(wT x)
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# CE
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
    X = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X.shape[1])
    
    theta, loss_history = gradient_descent(X, y, theta, alpha, iterations)
    
    return theta, loss_history

# Predict function
def predict(X, theta):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y_pred = hypothesis(X, theta)
    return (y_pred >= 0.5).astype(int)

# Function to plot decision boundary
def plot_decision_boundary(X, y, theta, title, is_xor=False):
    plt.figure(figsize=(6, 6))
    
    # Plot data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

    # Create grid to evaluate model
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

    if is_xor:
        # For XOR, use the extended feature space
        X_grid = np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel(), (xx1.ravel() * xx2.ravel())]
    else:
        # For AND/OR, use the original feature space
        X_grid = np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()]
    
    y_grid_pred = hypothesis(X_grid, theta).reshape(xx1.shape)
    
    # Plot decision boundary
    plt.contour(xx1, xx2, y_grid_pred, levels=[0.5], cmap="Greys", vmin=0, vmax=1)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Define datasets for AND, OR, XOR logic gates
def get_datasets():
    # Input features
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # AND
    y_and = np.array([0, 0, 0, 1])

    # OR
    y_or = np.array([0, 1, 1, 1])

    # XOR
    y_xor = np.array([0, 1, 1, 0])

    return X, y_and, y_or, y_xor

# Train and evaluate logistic regression models
def train_and_evaluate():
    X, y_and, y_or, y_xor = get_datasets()

    # Hyperparameters
    alpha = 0.1  # Learning rate
    iterations = 1000  # Number of iterations

    print("Training AND gate model...")
    theta_and, _ = logistic_regression(X, y_and, alpha, iterations)
    print(f"Trained parameters (theta) for AND gate: {theta_and}")
    predictions_and = predict(X, theta_and)
    print(f"AND gate predictions: {predictions_and}")
    print(f"AND gate actual labels: {y_and}\n")
    plot_decision_boundary(X, y_and, theta_and, "Decision Boundary for AND Gate")

    print("Training OR gate model...")
    theta_or, _ = logistic_regression(X, y_or, alpha, iterations)
    print(f"Trained parameters (theta) for OR gate: {theta_or}")
    predictions_or = predict(X, theta_or)
    print(f"OR gate predictions: {predictions_or}")
    print(f"OR gate actual labels: {y_or}\n")
    plot_decision_boundary(X, y_or, theta_or, "Decision Boundary for OR Gate")

    # For XOR, add interaction feature
    X_xor = np.c_[X, X[:, 0] * X[:, 1]]  # Add interaction term (x1 * x2)
    print("Training XOR gate model...")
    theta_xor, _ = logistic_regression(X_xor, y_xor, alpha, iterations)
    print(f"Trained parameters (theta) for XOR gate: {theta_xor}")
    predictions_xor = predict(X_xor, theta_xor)
    print(f"XOR gate predictions: {predictions_xor}")
    print(f"XOR gate actual labels: {y_xor}\n")
    plot_decision_boundary(X, y_xor, theta_xor, "Decision Boundary for XOR Gate", is_xor=True)

# Run training and evaluation
if __name__ == "__main__":
    train_and_evaluate()
