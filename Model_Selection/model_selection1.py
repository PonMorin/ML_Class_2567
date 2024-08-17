import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

# Generate data
np.random.seed(0)
X = np.linspace(0, 2*np.pi, 20)
# y = np.sin(X) + np.random.normal(0, 0, 100)
y = np.sin(X) + np.random.normal(0, 0.3, 20)

# Split the data into training and testing sets
# X_train, X_test = X[:80], X[80:]
# y_train, y_test = y[:80], y[80:]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=.8)
# Fit and plot models of different degrees

degrees = [1, 3, 8]
splits = [10]

# Create polynomial features
plt.figure(figsize=(8, 8))
for k_flod in splits:
    for i, degree in enumerate(degrees, 1):
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression)
        ])

        # Train model
        pipeline.fit(X_train.reshape(-1, 1), y_train)

        # Predictions
        y_train_pred = pipeline.predict(X_train.reshape(-1, 1))
        y_test_pred = pipeline.predict(X_test.reshape(-1, 1))

        # Compute errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        # Get coefficients
        coefficients = linear_regression.coef_

        # Cross-Validation scores
        cv_scores = cross_val_score(pipeline, X_train.reshape(-1, 1), y_train, cv=k_flod, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        # Plotting

        print('\n')
        print('---------------------------------------------------------------------------------------------------------------------------------')
        print(f'When Degrees-->{degree} Cross validation-->{k_flod}')
        print(f'Polynomial Regression (Degree = {degree})\nTrain RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}\nCV RMSE: {cv_rmse.mean():.2f} (± {cv_rmse.std():.2f})')
        print(f'Coefficients for degree {degree}: {coefficients}')
        print(f'Cross-Validation RMSE for degree {degree}: {cv_rmse.mean():.2f} (± {cv_rmse.std():.2f})')

        plt.subplot(2, 2, i)
        plt.scatter(X_train, y_train, color='blue', label='Training data',alpha=0.2)
        plt.scatter(X_test, y_test, color='red', label='Testing data',alpha=0.2)
        plt.plot(X, pipeline.predict(X.reshape(-1, 1)), color='green', label='Model')
        plt.title(f'Polynomial Regression (Degree = {degree})\nTrain RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}\nCV RMSE: {cv_rmse.mean():.2f} (± {cv_rmse.std():.2f})')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()

    plt.tight_layout()
    plt.show()
# Print coefficients and CV RMSE


