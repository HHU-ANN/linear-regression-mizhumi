# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np
def rg_rs(x,y,a):
    I=np.identity(x.shape[1])
    w=np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + a* I), x.T), y)
    return w
def ls_rs(x,y,a=0.1,lt=1e-12):
    _,n = x.shape
    # print(n)
    w = np.zeros(n)
    min = 1e10
    for i in range(10000):
        l1 = a * (np.sum(np.abs(w)))
        mse = np.sum(((x @ w) - y.T) @ ((x @ w) - y.T).T) / (np.shape(x)[0])
        ls_ls = mse + l1
        dw = x.T @ ((x @ w) - y.T) + a * np.sign(w)
        w = w - lt * dw
        if np.abs(min - ls_ls) < 0.0001:
            break
        if min >= ls_ls:
            min = ls_ls
            b = w
    return b

def ridge(data):
    x, y = read_data()
    a = 1e-12
    w = rg_rs(x, y, a)
    return np.dot(data, w)
    
def lasso(data):
    x, y = read_data()
    w = ls_rs(x, y)
    return data @ w[:6] + w[-1]

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y