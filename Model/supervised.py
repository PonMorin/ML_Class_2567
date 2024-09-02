import numpy as np
import matplotlib.pyplot as plt

class Regression():
    def __init__(self, weights=np.random.rand()*10, biases=np.random.rand(), learning_rate=.01, iterations=10000, verbose=False):
        """
        Parameters:
        Returns:
        """
        self.weights = weights
        self.biases = biases
        self.lr = learning_rate
        self.iterations = iterations
        if verbose:
            print(f"Weights of this model = {self.weights}")
            print(f"Biases of this model = {self.biases}")
            print(f"Learning rate of this model = {self.lr}")
            print(f"Iteration of this model = {self.iterations}")

    def Prediction(self, X):
        return np.dot(X, self.weights) + self.biases
    
    def Hypothesis(self, X):
        return np.dot(X, self.weights) + self.biases + np.random.random()
    
    # Standardization
    def Z_score(self, X):
        mean_X = np.array([np.mean(X)])
        standard_deviation = np.array([np.std(X)])
        x_standard = (X - mean_X)/standard_deviation
        return x_standard
    
    # Normalization
    def Min_Max_scaling(Self, X):
        return (X - np.min(X))/(np.max(X) - np.min(X))

    
    def Forward(self, X, y, W):
        """
        Parameters:
        X (array) : Independent Features
        y (array) : Dependent Features/ Target Variable
        W (array) : Weights 

        Returns:
        loss (float) : Calculated Sqaured Error Loss for y and y_pred
        y_pred (array) : Predicted Target Variable
        """
        y_pred = np.sum(W * X)
        loss = ((y_pred-y)**2)/2
        return loss, y_pred

    def Mean_absolute_error(self, X, y, verbose=False):
        """
        Parameters:
        Returns:
        """
        length = len(y)
        predictions = self.Prediction(X=X)
        if verbose:
            print(f"Prediction from data X: {predictions}")
        accumulated_error = 0.0
        for predict, target in zip(predictions, y):
            accumulated_error += np.abs(predict - target)
            if verbose:
                print(f"|{predict} - {target}| = {accumulated_error}")
        mae_error = (1.0/length)*accumulated_error
        return mae_error

    def Mean_squared_error(self, X, y):
        """
        Parameters:
        Returns:
        """
        predictions = self.Prediction(X=X) # Get predictions using current w0 and w1
        squareErrorSum = 0
        for i in range(len(y)):
            error = predictions[i] - y[i] # Calculate error for each data point
            squareErrorSum += error ** 2  # Square the error and add to the sum
        mse = (1 / (2 * len(y))) * squareErrorSum  # Calculate MSE by averaging squared errors
        return mse
        
    def Root_mean_squared_error(self, X, y, n):
        """
        Parameters:
        Returns:
        """
        return np.sqrt(self.Mean_squared_error(X, y, n))
        # residual
    
    def Cost_function(self, X, y, y_predict):
        """
        Parameters:
        Returns:
        """
        error = y_predict - y
    
    def Cost_function_algebra(self, X, y, w, b):
        """
        Parameters:
        Returns:
        """
        n = len(y)
        cost = 0
        for i in range(n):
            y_pred = w + b * X[i]
            cost += (y[i] - y_pred) ** 2
        totalcost = (1/2*n)*cost
        return totalcost
    
    def Sum_of_squared_error(self, X, y_true):
        """
        Parameters:
        Returns:
        """
        Y_prediction = self.Hypothesis(X)
        sse = np.sum((y_true - Y_prediction)**2)
        return sse
    
    def R_score(self, y, y_predict):
        """
        Parameters:
        Returns:
        """
        SSres = np.square(y - y_predict)
        SStot = np.square(y - np.mean(y))
        R = 1-(SSres/SStot)
        return R

    def Sigmoid_activation(self, x):
        """
        Parameters:
        Returns:
        """
        return 1/(1+np.exp(x))
        
    
    def Feature_scaling(self):
        """
        Parameters:
        Returns:
        """
        pass
    
    def Gradient_descent_algebra(self, X, y, verbose=False):
        """
        Parameters
        w : theta1 or weight
        b : theta0 or bias
        """
        
        weight_history = []
        bias_history = []
        mse_history = []
        length = len(y)
        for i in range(self.iterations):
            hypothesis = self.Prediction(X)
            self.weights -= self.lr * (1/length) * np.sum((hypothesis - y)*X)
            self.biases  -= self.lr * (1/length) * np.sum(hypothesis - y)
            if verbose:
                print(f"weight change {i} -> {self.weights}")
                print(f"bias change   {i} -> {self.biases}")            
            mse = self.Mean_squared_error(X, y)
            weight_history.append(self.weights)
            bias_history.append(self.biases)
            mse_history.append(mse)
        return self.weights, self.biases, mse, weight_history, bias_history, mse_history
    
    def Gradient_descent(self, X):
        """
        Parameters:
        Returns:
        """
        pass
    
    def RSS_function(y, y_predict, lamda, theta):
        """
        Parameters:
        y (array) :
        y_predict :
        lamda
        theta : weight
        Returns:
        """
        error = y_predict - y
        rss = np.dot(error.T, error) + ((lamda) * np.dot(theta[-1].T, theta[-1]))
        return rss
    # Regularization = Loss Function + Penalty
    def Ridge_regression():
        """
        Parameters:

        Returns:
        """
        pass

    def Lasso_regression():
        """
        Parameters:
        Returns:
        """
        pass

    def Elastic_net_regression():
        """
        Parameters:
        Returns:
        """
        pass

    def Normal_equation(self, X, y):
        """
        Parameters:
        Returns:
        """
        return np.dot(np.linalg.inv(np.dot(X.T, X)),np.dot(X.T, y))
    
    def Update_weights(self, X, y, y_predict=0):
        """
        Parameters:
        Returns:
        """
        if y_predict:
            error = y_predict - y
        else:
            error = self.Prediction(y) - y
        m = X.shape[0]
        # new_weight = old_weight - ((self.lr/n) * (np.dot(X.T, error)))
        dw = -(X.T).dot(error)/m
        db = -np.sum(error)/m
        self.weights = self.weights - self.lr * dw
        self.biases = self.biases - self.lr * db
        # new_bias = old_bias - ((lr/ n))
        return self.weights, self.biases
    
    def Split_data(self, X, y, train_size = 0.8, test_size=0.2, validation_size=0.3, random_state=0, verbose=False):
        """
        Parameters:
        Returns:
        """
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        X_validation = []
        y_validation = []
        np.random.seed(random_state)                 #set the seed for reproducible results
        indices = np.random.permutation(len(X))      #shuffling the indices
        data_train_size = int(X.shape[0] * train_size) #Get the train size
        data_test_size = int(X.shape[0] * test_size) #Get the test size
        data_validation_size = int(X.shape[0] * validation_size) #Get the validation size 
        X_train = X[indices]
        y_train = y[indices]
        X_test = X[indices]
        y_test = y[indices]
        X_validation = X[indices]
        y_validation = y[indices]
        return X_train, y_train, X_test, y_test, X_validation, y_validation

    def Train_regression_model(self, X, y, epochs=100, verbose=False):
        """
        Parameters:
        Returns:
        """
        cost_history = [] # Train cost
        loss_history = [] # Train loss

        # self.Cost_function_algebra()
        for i in range(self.iterations):
            weight, bias = self.Update_weights(X=X, y=y)

        return weight, bias
    
    def Plot_test(self, X, y, y_plot, verbose=False):
        """
        Parameters:
        Returns:
        """
        weights, biases, mse, weight_history, bias_history, mse_history = self.Gradient_descent_algebra(X=X, y=y, verbose=verbose)
        print(f'Optimize weight: {weights:.5f}')
        print(f'Optimize bias  : {biases:.5f}')
        print(f'Optimize MSE   : {mse:.5f}')
        plt.xlabel(u'X', fontname='Tahoma')
        plt.ylabel(u'Y', fontname='Tahoma')
        plt.scatter(X, y_plot)
        plt.plot(X, weights * X + biases)
        plt.plot(X, y)
        plt.show()

class Generalization():
    def __init__(self, learning_rate=.01, iterations=10000, verbose=False):
        """
        Parameters:
        Returns:
        """
        self.lr = learning_rate
        self.iterations = iterations
        if verbose:
            print(f"Learning rate of this model = {self.lr}")
            print(f"Iteration of this model = {self.iterations}")

    def Sin_function(self, X):
        """
        Parameters:
        Returns:
        """
        return np.sin(np.dot(np.pi ,X))
    
    def Constant_model(self, X):
        """
        Parameters:
        Returns:
        """
        return np.mean(np.sin(np.dot(np.pi, X)))
    
    def Init_theta(shape):
        """
        Parameters:
        Returns:
        """
        theta = np.array(np.zeros(shape))
        return theta
    
    def Cost_function(n, Y, Y_pred):
        """
        Parameters:
        Returns:
        """
        error = Y_pred - Y
        cost = (1/2*n) * np.dot(error.T, error)
        return cost

    def Liner_model(self, X, theta):
        """
        Parameters:
        Returns:
        """
        return np.dot(X, theta)
    
    def Normal_equation(self, X, y):
        """
        Parameters:
        Returns:
        """
        return np.dot(np.linalg.pinv(np.dot(X.T, X)),np.dot(X.T, y))
    
    def Mean_model(self, mean_model):
        """
        Parameters:
        Returns:
        """
        return np.mean(mean_model, axis=0)

class Classification():
    def __init__(self, learning_rate=.01, iterations=10000, verbose=False):
        """
        Parameters:
        Returns:
        """
        self.lr = learning_rate
        self.iterations = iterations
        if verbose:
            print(f"Learning rate of this model = {self.lr}")
            print(f"Iteration of this model = {self.iterations}")

    def Sigmoid_function(self, z):
        """
        Parameters:
        z:
        Returns:

        """
        return 1/(1 + np.exp(np.negative(z)))
    
    def Softmax_function(self, z):
        """
        Parameters:
        Returns:
        """
        e = np.exp(z)
        return e/e.sum()
    
    def standardization(self, X):
        mean_x = np.mean(X)
        std_x = np.std(X)
        X_sd = (X - mean_x)/std_x
        return X_sd
    
    def F1_score(self, y, y_pred):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 0:
                tp += 1
            elif y[i] == 1 and y_pred[i] == 0:
                fn += 1
            elif y[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y[i] == 0 and y_pred[i] == 0:
                tp += 1
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        f1_score = 2*precision * recall / (precision + recall)
        return f1_score
    
    def init_theta(self, X):
        theta = np.zeros(X.shape[1])
        return theta
    
    def logistic_model(self, X, theta):
        z = np.dot(X, theta.T)
        y_pred = self.Sigmoid_function(z)
        return y_pred
    
    def cost_function(self, y, y_pred):
        m = len(y)
        cost = - (1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost 
    
    def update_weight(self, n, old_weight, X, y, y_pred, lr=0.2):
        error = y_pred - y
        new_weight = old_weight - ( (lr/n) * np.dot(X.T, error))
        return new_weight

    def gradient_descent(self, n, X, y, theta, steps):
        cost_history = []
        # theta_list = []
        for _ in range(steps):
            Y_pred = self.logistic_model(X, theta)
            cost = self.cost_function(y, Y_pred)
            cost_history.append(cost)
            theta = self.update_weight(n, theta, X, y, Y_pred)
            # theta_list.append(theta)

        return cost_history, theta

class Supervised_model():
    def __init__(self):
        pass

class NeuralNetwork():
    def __init__(self, weights, biases, verbose=False):
        """
        Parameters:
        Returns:
        """
        self.weights = []
        self.biases = []
        self.threshold = 0.5

        self.weights = weights
        self.biases = biases
        if verbose:
            print(f'Weights = {self.weights}')
            print(f'Biases = {self.biases}')

if __name__ == "__main__":
    G = Generalization()
    X = np.linspace(-1, 1)
    X_b = np.c_[np.ones((len(X), 1)), X]
    y = G.Sin_function(X=X)
    steps = 1000

    for i in range(steps):
        Data = np.random.choice(X, 2)
        y_sample = G.Sin_function(Data)
        print(y_sample)
        X_b_sample = np.c_[np.ones((len(Data), 1)), Data]
        n = len(Data)


    plt.plot(X, y, c="#4CAF50")
    plt.plot()
    plt.show()
"""
# X = np.arange(1, 31, 2)
X = np.arange(1, 21)
# y = 2.5+X*w+np.random.randn(15)*0.5
# y = 3+X*2
print(X)
# y = [3.69341451, 3.1776988, 5.16130142, 6.73462437, 7.63179596, 
#      7.95533464, 9.00517668, 10.37146896, 11.44725888, 12.28205542, 
#      12.76848293, 14.18088657, 14.76103989, 16.09339126, 17.23117329]

y_true = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5]
# y_pred = [1.78076574, 3.60684733, 3.65545, 3.93909785, 5.79271944, 5.72580876, 5.54748533, 5.67516857, 7.17979059, 7.50852625, 7.43362042, 8.57904178, 10.2732663, 9.8621399, 9.8643251, 10.10607952, 10.44289345, 12.06076674, 11.96533227, 12.4975351]
y_plot = [1.78076574, 3.60684733, 3.65545, 3.93909785, 5.79271944, 5.72580876, 5.54748533, 5.67516857, 7.17979059, 7.50852625, 7.43362042, 8.57904178, 10.2732663, 9.8621399, 9.8643251, 10.10607952, 10.44289345, 12.06076674, 11.96533227, 12.4975351]
w = 0
b = 0

#Independent Feature Scaling
Xs = X
Xs = (Xs - int(np.mean(Xs)))/np.std(Xs)
#Dependent Feature Scaling
ys = y_true
ys = (ys - np.mean(ys))/np.std(ys)

# X_b = np.c_[np.ones((len(Xs), 1)), Xs]

# X_b = np.c_[np.ones((len(X), 1)), X]
# print(neuron.Normal_equation(X=X_b, y=y_true))

neuron = Regression(weights=w, biases=b, learning_rate=0.001, iterations=10000, verbose=False)
neuron.Train_regression_model(X=X,y=y_true,epochs=10)
weights, biases, mse, weight_history, bias_history, mse_history = neuron.Gradient_descent_algebra(X=X, y=y_true, verbose=False)
y_predict = neuron.Hypothesis(X)
print(y_predict)
# y_predict = neuron.Prediction(X)
# print(y_predict)

neuron.Plot_test(X=X, y=y_predict, y_plot=y_plot)
# print(neuron.Mean_absolute_error(X=X, y=y_pred, y=y_true, verbose=False))
"""