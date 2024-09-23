import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_classification, make_moons
from scipy.stats import norm
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

X, y = make_moons(n_samples=400, noise=0.2, random_state=42)

train_size=int(0.75*X.shape[0])
test_size=int(0.25*X.shape[0])

print("Training set size : "+ str(train_size))
print("Testing set size : "+str(test_size))
sc = StandardScaler()
X = sc.fit_transform(X)

X_train=X[0:train_size,:]
y_train=y[0:train_size]
#testing set split 
X_test=X[train_size:,:] 
y_test=y[train_size:]

nb = GaussianNB()
nb.fit(X_train,y_train)

def generate_data(class_data_dic,X_train,y_train):
    first_one=True
    first_zero=True
    for i in range(y_train.shape[0]):
        X_temp=X_train[i,:].reshape(X_train[i,:].shape[0],1)
        if y_train[i]==1:
            if first_one==True:
                class_data_dic[1]=X_temp
                first_one=False
            else: 
                class_data_dic[1]=np.append(class_data_dic[1],X_temp,axis=1)                                      
        elif y_train[i]==0:
            if first_zero==True:
                class_data_dic[0]=X_temp
                first_zero=False
            else:
                class_data_dic[0]=np.append(class_data_dic[0],X_temp,axis=1)   
    return class_data_dic

class_data_dic = {}
class_data_dic = generate_data(class_data_dic=class_data_dic,X_train=X_train, y_train=y_train)

mean_0=np.mean(class_data_dic[0],axis=0)
mean_1=np.mean(class_data_dic[1],axis=0)        
std_0=np.std(class_data_dic[0],axis=0)       
std_1=np.std(class_data_dic[1],axis=0)                  

def qda_discriminant(x, mean, cov, prior):
    inv_cov = np.linalg.inv(cov)
    log_det_cov = np.log(np.linalg.det(cov))
    return -0.5 * np.dot(np.dot((x - mean).T, inv_cov), (x - mean)) - 0.5 * log_det_cov + np.log(prior)

# def qda_predict(X):
#     discriminants = Discriminants(X)
#     return np.argmax(discriminants, axis=1)

# def Discriminants(X):
#     discriminants = np.array([
#         [qda_discriminant(x, means[k], covariances[k], priors[k]) for k in range(len(means))]
#         for x in X
#     ])
#     return discriminants

# def qda_posterior(X):
#     discriminants = Discriminants(X)
#     max_discriminants = np.max(discriminants, axis=1, keepdims=True)
#     exp_discriminants = np.exp(discriminants - max_discriminants)
#     posteriors = exp_discriminants / np.sum(exp_discriminants, axis=1, keepdims=True)
#     return posteriors

def likelyhood(x,mean,sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))

def posterior(X,X_train_class,mean_,std_):
    product=np.prod(likelyhood(X,mean_,std_),axis=1)
    product=product*(X_train_class.shape[0]/X_train.shape[0])
    return product

p_1=posterior(X_test.reshape(-1, 1),class_data_dic[1],mean_1,std_1)   
p_0=posterior(X_test.reshape(-1, 1),class_data_dic[0],mean_0,std_0)
y_pred=1*(p_1>p_0)

# #visualize the training set 
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j,marker='.')
# plt.title('Training set')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

from matplotlib.colors import ListedColormap
# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
plt.contourf(X1, X2, nb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('orange', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j,marker='.')
plt.title('Naive Bayes Classification our implementation(Training set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('orange', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j,marker='.')
plt.title('Naive Bayes Classification scikit-learn (Test set)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()