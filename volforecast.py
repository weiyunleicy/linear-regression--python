import pandas as pd
import numpy as np
#读取数据
dataset = pd.read_csv('voldata.csv')

x1=[1,1,1,1,1,1,1,1]
x2=[1,2,3,4,5,6,7,8]
xdata1=np.c_[np.transpose(x1),np.transpose(x2)]

Y = np.transpose(dataset.iloc[:,1:])
#设定起始theta和学习率
theta_init = np.zeros([xdata1.shape[1],dataset.shape[0]])
alpha = 0.01
'''
#输入数据标准化.根据实际feature数据选择是否标准化
def minmax_normalize(X):  # 最小-最大标准化
    x_mean = np.mean(X, axis=0)
    x_range = np.max(X, axis=0) - np.min(X, axis=0)
    x_norm = (X - x_mean) / x_range
    return x_mean, x_range, x_norm

X_mean,X_range,X = minmax_normalize(xdata1)
'''
#代价计算
def compute_cost(x,y,theta):
    hypthesis = np.dot(x,theta)
    error = hypthesis-y
    cost = (error**2).sum(axis = 0)
    cost = cost/(2*x.shape[0])
    return cost
#梯度下降
def gradientDescent(x,y,theta_init,alpha,iter_num):
    theta = theta_init
    J_history = np.zeros([iter_num,Y.shape[1]])
    for num in range(iter_num):
        J_history[num] = compute_cost(x,y,theta)
        hyp = np.dot(x,theta)
        theta = theta -  np.transpose(alpha*np.dot(np.transpose(hyp-y),x)/x.shape[0])
    return theta,J_history
#开始训练
theta,J_history = gradientDescent(xdata1,Y,theta_init,0.01,2000)

#预测
x_predict = np.array([[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15]])
Y_predict = np.dot(x_predict,theta)

Y_predict = np.transpose(Y_predict)
Y_final = pd.DataFrame(np.c_[dataset.iloc[:,0],Y_predict])#矩阵转为表格
Y_final.columns = ['City','2018','2019','2020','2021','2022','2023','2024','2025'] #修改列名
Y_final.to_csv('predict_20192022.csv', index = None)