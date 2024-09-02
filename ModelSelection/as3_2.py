import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Define the model and pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.001, 0.01, 0.1]
}

# Outer loop: KFold cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner loop: GridSearchCV for hyperparameter tuning
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='accuracy')

# Perform Nested Cross-Validation
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='accuracy')

print(f'Nested CV accuracy: {nested_scores.mean():.2f} ± {nested_scores.std():.2f}')

# Cross-Validation แบบลูปเดียว
single_cv = GridSearchCV(pipeline, param_grid, cv=outer_cv, scoring='accuracy')
single_cv.fit(X, y)
single_cv_score = single_cv.best_score_

print(f'Single CV accuracy: {single_cv_score:.2f}')

# Plotting the comparison
labels = ['Single CV', 'Nested CV']
scores = [single_cv_score, nested_scores.mean()]
errors = [0, nested_scores.std()]

plt.figure(figsize=(10, 6))
plt.bar(labels, scores, yerr=errors, color=['blue', 'green'], capsize=10)
plt.xlabel('Cross-Validation Method')
plt.ylabel('Accuracy')
plt.title('Comparison of Single Loop and Nested Cross-Validation')
plt.ylim(0, 1)
plt.show()
