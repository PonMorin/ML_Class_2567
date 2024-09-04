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

def standardization(X):
    mean_x = np.array(np.mean(X))
    std_x = np.array(np.std(X))
    X_sd = (X - mean_x) / std_x
    return X_sd

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_model(X, theta):
    z = np.dot(X, theta.T)
    y_pred = Sigmoid(z)
    return y_pred

df = sns.load_dataset('iris')
X = df.drop('species', axis=1)
y, class_name = pd.factorize(df.species, sort=True)

sc = StandardScaler()
X_sc = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.25, random_state=1)
data = np.array(df)
species = [0, 0, 0]

likelihood = 0
posterior = 0
Event = 0 # H
Evidence = 0 # E

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
print(species)

# print(df.keys)
# print(df['sepal_length'])
# print(df['sepal_width'])
# print(df['petal_length'])
# print(df['petal_width'])
# print(df['species'])

sns.FacetGrid(df, hue="species", height=6).map(plt.scatter, 'sepal_length', 'sepal_width').add_legend()
plt.show()




