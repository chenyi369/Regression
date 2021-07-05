import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_xlsx(path):
    data = pd.read_excel(path)
    print(data)
    return data

def MinMaxScaler(data):
    col = data.shape[1]
    for i in range(0, col-1):
        arr = data.iloc[:, i]
        arr = np.array(arr)
        min = np.min(arr)
        max = np.max(arr)
        arr = (arr-min)/(max-min)
        data.iloc[:, i] = arr
    return data

def train_test_split(data, test_size=0.2, random_state=None):
    col = data.shape[1]
    x = data.iloc[:, 0:col-1]
    y = data.iloc[:, -1]
    x = np.array(x)
    y = np.array(y)
    # 设置随机种子，当随机种子非空时，将锁定随机数
    if random_state:
        np.random.seed(random_state)
        # 将样本集的索引值进行随机打乱
        # permutation随机生成0-len(data)随机序列
    shuffle_indexs = np.random.permutation(len(x))
    # 提取位于样本集中20%的那个索引值
    test_size = int(len(x) * test_size)
    # 将随机打乱的20%的索引值赋值给测试索引
    test_indexs = shuffle_indexs[:test_size]
    # 将随机打乱的80%的索引值赋值给训练索引
    train_indexs = shuffle_indexs[test_size:]
    # 根据索引提取训练集和测试集
    x_train = x[train_indexs]
    y_train = y[train_indexs]
    x_test = x[test_indexs]
    y_test = y[test_indexs]
    # 将切分好的数据集返回出去
    # print(y_train)
    return x_train, x_test, y_train, y_test

def costFunction(x, y, theta):
    m = len(x)
    J = np.sum(np.power(np.dot(x, theta) - y, 2)) / (2 * m)
    return J

def gradeDesc(x,y,alpha=0.01,iter_num=2000):
    x = np.array(x)
    y = np.array(y).reshape(-1, 1)
    m = x.shape[0]
    n = x.shape[1]
    theta = np.zeros(n + 1).reshape(-1, 1)
    c = np.ones(m).transpose() #构建m行1列 x0=1
    x = np.insert(x, 0, values=c, axis=1)
    costs = np.zeros(iter_num)   # 初始化cost, np.zero生成1行iter_num列都是0的矩阵
    for i in range(iter_num):
        for j in range(n):
            theta[j] = theta[j] + np.sum((y - np.dot(x, theta)) * x[:, j].reshape(-1, 1)) * alpha / m
        costs[i] = costFunction(x, y, theta)
    return theta, costs

    # L2正则化
def l2costFunction(x, y, lamda, theta):
    m = len(x)
    J = np.sum(np.power((np.dot(x, theta) - y), 2)) / (2 * m) + lamda * np.sum(np.power(theta, 2))
    return J

def l2gradeDesc(x, y, alpha, iter_num, lamda):
    x = np.array(x)
    y = np.array(y).reshape(-1, 1)
    m = x.shape[0]
    n = x.shape[1]
    theta = np.zeros(n + 1).reshape(-1, 1)
    c = np.ones(m).transpose()
    x = np.insert(x, 0, values=c, axis=1)
    costs = np.ones(iter_num)
    for i in range(iter_num):
        for j in range(n):
            theta[j] = theta[j] + np.sum((y - np.dot(x, theta)) * x[:, j].reshape(-1, 1)) * (alpha / m) - 2 * lamda * theta[j]
        costs[i] = l2costFunction(x, y, lamda, theta)
    return theta, costs

def predict(x, theta):
    x = np.array(x)
    c = np.ones(x.shape[0]).transpose()
    x = np.insert(x, 0, values=c, axis=1)
    y = np.dot(x, theta)
    return y

def mse(y_true, y_test):
    mse = np.sum(np.power(y_true - y_test, 2)) / len(y_true)
    return mse


if __name__ == '__main__':
    data = read_xlsx(r'D:\数据集\regression.xlsx')
    scaler_data = MinMaxScaler(data)
    x_train, x_test, y_train, y_test = train_test_split(scaler_data)

    #梯度下降
    # theta, costs = gradeDesc(x_train, y_train, alpha=0.01, iter_num=2000)
    # y_pred = predict(x_test,theta)
    # print(y_pred)
    # print('mse=', mse(y_test, y_pred))

    #带L2正则化的梯度下降
    theta, costs = l2gradeDesc(x_train, y_train, alpha=0.01, iter_num=2000, lamda=0.001)
    print(theta)
    y_pred = predict(x_test,theta)
    print(y_pred)
    print('mse=', mse(y_test, y_pred))

    #画图cost曲线
    ax1 = plt.subplot(121)
    iter_num = 2000
    x_ = np.linspace(1, iter_num, iter_num)
    ax1.plot(x_, costs)
    plt.show()

