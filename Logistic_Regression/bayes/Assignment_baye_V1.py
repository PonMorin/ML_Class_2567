"""
1.เขียนโปรแกรมสำหรับสร้างตัวจำแนกแบบเบส์สำหรับการแจกแจงปรกติตัวแปรเดียว กรณีที่ความแปรปรวนของทั้งสองคลาสเท่ากัน
2.เขียนโปรแกรมสำหรับสร้างตัวจำแนกแบบเบส์สำหรับการแจกแจงปรกติตัวแปรเดียว กรณีที่ความแปรปรวนของทั้งสองคลาสไม่เท่ากัน
3.เขียนโปรแกรมสำหรับสร้างตัวจำแนกกำลังสอง>
4.เขียนโปรแกรมสำหรับสร้างตัวจำแนกเชิงเส้น
ทั้ง 4 ข้อให้ วาดกราฟ likelihood, posterior และขอบตัดสินใจ โดยทำสองรูปแบบ คือ - กำหนดค่าพารามิเตอร์ของการแจกแจก - สุ่มตัวอย่างเพื่อนำมาคำนวณค่าพารามิเตอร์ของการแจกแจง 
5.เขียนโปรแกรมสำหรับ plot decision boundary เปรียบเทียบระหว่าง LDA, QDA และ Logistic regression โดยอาจจะมีการเพิ่มพจน์ second order polynomial 
โดยอาจจะใช้การสุ่มข้อมูลในรูปแบบต่างๆดังนี้ https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py หรือ https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle®Dataset=reg-plane&learningRate=0.03®ularizationRate=0&noise=0&networkShape=4,2&seed=0.87693&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Baye_Decision():
    def __init__(self):
        pass

    def naive_bayes_equation(self, TYPE=1):
        if(TYPE):
            print('\n')
            print('         P(E|H) P(H)')
            print('P(H|E) = -----------')
            print('             P(E)   ')
            print('                    ')
            print('H = Event')
            print('E = Evidence')
            print('Pr(H|E) = Posterior Probability')
            print('Pr(E|H) = Likelihood')
            print('Pr(H)   = Prior Probability')
            print('Pr(E)   = Predictor Prior Probability')
            print('\n')
        else:
            print('\n')
            print('         P(x|c) P(c)')
            print('P(c|x) = -----------')
            print('             P(x)   ')
            print('                    ')
            print('c = Class')
            print('x = Attribute')
            print('P(c|x) = Posterior Probability')
            print('P(x|c) = Likelihood')
            print('P(c)   = Class Prior Probability')
            print('P(x)   = Predictor Prior Probability')
            print('\n')
    def naive_bayes(self):
        pass

        


def standardization(X):
    mean_x = np.array(np.mean(X))
    std_x = np.array(np.std(X))
    X_sd = (X - mean_x) / std_x
    return X_sd

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def mean_value(n, X):
    return(1/n)*np.sum(X)

def standard_deviation(n, X):
    mean = np.sum(X)/n
    # print(mean)
    # variance = sum([((x-mean)**2) for x in X])/n
    variance = sum([((x-mean)**2) for x in X])/(n-1)
    # print(variance)
    result = variance ** 0.5
    # print(result**2)
    return result

def density_function(mean, X, std):
    # print(mean)
    # return (1/(np.sqrt(2*np.pi))*std)*np.exp(np.negative((X - mean)**2/(2*std**2)))
    return np.exp(np.negative((X - mean)**2/(2*std**2)))/((np.sqrt(2*np.pi))*std)

def normal_distribution(n, X, std):
    mean = np.sum(X)/n
    variance = sum([((x-mean)**2) for x in X])/n
    std = variance ** 0.5
    return np.exp(np.negative(0.5*(np.square((X - mean)/std))))/((np.sqrt(2*np.pi))*std)

def init_theta(X):
    theta = np.array(np.zeros(X))
    return theta

def logistic_model(X, theta):
    z = np.dot(X, theta)
    y_pred = Sigmoid(z)
    return y_pred

def cost_function(y, y_pred):
    cost = np.dot(-y.T, np.log(y_pred)) - np.dot((1-y).T, np.log(1 - y_pred))
    return cost 

def update_weight(n, old_weight, X, y, y_pred, lr=0.2):
    error = y_pred - y
    new_weight = old_weight - ( (lr/n) * (np.dot(X.T, error)) )
    return new_weight

def density_function(mean, X, std):
    # print(mean)
    # return (1/(np.sqrt(2*np.pi))*std)*np.exp(np.negative((X - mean)**2/(2*std**2)))
    return np.exp(np.negative((X - mean)**2/(2*std**2)))/((np.sqrt(2*np.pi))*std)

def normal_distribution(n, X, std):
    mean = np.sum(X)/n
    variance = sum([((x-mean)**2) for x in X])/n
    std = variance ** 0.5
    return np.exp(np.negative(0.5*(np.square((X - mean)/std))))/((np.sqrt(2*np.pi))*std)

def single_threshold(n, X, Class):
    mean = np.sum(X)/n
    variance = sum([((x-mean)**2) for x in X])/n
    std = variance ** 0.5
    # print(X-mean)
    return np.negative(np.square(X - mean)/2*variance) - 0.5*np.log(2*np.pi) - np.log(std) - np.log(normal_distribution(len(Class), Class, np.std(Class))) - np.log(normal_distribution(len(X), X, np.std(X)))

def linear_discriminant(mu_c1, mu_c2, Sigma):
    Sigma_inv = np.linalg.inv(Sigma)
    print(Sigma_inv)
    w = np.dot(Sigma_inv, mu_c1 - mu_c2)
    print(w)
    b = -0.5 * np.dot(np.dot(mu_c1, Sigma_inv), mu_c1) + 0.5 * np.dot(np.dot(mu_c2, Sigma_inv), mu_c2)
    print(b)
    return w, b

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

if __name__ == '__main__':

    df = sns.load_dataset('iris')
    X = df.drop('species', axis=1)
    y, class_name = pd.factorize(df.species, sort=True)

    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.25, random_state=1)
    data = np.array(df)
    species = [0, 0, 0]

    # print(len(df))
    # def Discriminant():

    # print(math.log(1))

    for data in df['species']:
        if data == 'setosa':
            species[0] += 1
        elif data == 'versicolor':
            species[1] += 1
        elif data == 'virginica':
            species[2] += 1
    # print(species)
    # print(len(df['species']))
    
    test_list = [4, 5, 8, 9, 10] 
    ary = [2, 3, 4, 5, 6]
    # ary = np.array([[1, 2, 3, 2], [4, 5, 6, 5],])
    # print(density_function(len(ary), ary, np.std(ary)))
    # print(density_function(3, 2, 4))
    print(normal_distribution(len(ary), ary, np.std(X)))
    
    # print(ary.shape)
    
    # model = Baye_Decision()
    # model.naive_bayes()
    # print(df.keys)
    # print(df['sepal_length'])
    # print(df['sepal_width'])
    # print(df['petal_length'])
    # print(df['petal_width'])
    # print(df['species'])

    # sns.FacetGrid(df, hue="species", height=6).map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()
    # plt.show()

    

# likelihood = [0,0]
# prior = [0,0]
# posterior = 0
# event = 0 # H
# evidence = 0 # E
# rate = 0
# A = 0.0
# B = 0.0

# try:
#     while True:
#         rate = int(input("What Rate you have A or B => 1,2 respective: "))
#         if rate == 1:
#             A = float(input("Enter probability percent of A: "))
#             if 0 <= A <= 1:
#                 B = 1 - A
#                 break
#             else:
#                 print("invalid input that must be float in between 0-1")
#         elif rate == 2:
#             B = float(input("Enter probability percent of B: "))
#             if 0 <= B <= 1:
#                 A = 1 - B
#                 break
#             else:
#                 print("invalid input that must be float in between 0-1")
#         else:
#             print("You incorrect input try again: ")
    
#     likelihood = float(input("Enter number of : "))


#     posterior = (likelihood * A)/B
#     # posterior = (likelihood*prior)/evidence
   
#     print("p(F = a|B = r) = ",Apple[0]/totalred)
#     print("p(F = o|B = r) = ",Apple[1]/totalblue)
#     print("p(F = a|B = b) = ",Orange[0]/totalred)
#     print("p(F = o|B = b) = ",Orange[1]/totalblue)
#     print("Apple = %.2f"%(apple_red + apple_blue))
#     print("Orange = %.2f"%(orange_red + orange_blue))
#     print("Apple from red box rate = %.2f"%(apple_red/fruit_apple))
#     print("Apple from red box rate = %.2f"%(apple_blue/fruit_apple))
#     print("Orange from red box rate = %.2f"%(orange_red/fruit_orange))
#     print("Orange from red box rate = %.2f"%(orange_blue/fruit_orange))
# except ValueError:
#     print("Error Invalid input Type")
# except ZeroDivisionError:
#     print("Error Zero Divisoion")
