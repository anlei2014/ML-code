# -*- coding: utf-8 -*- 
# @Time : 18/11/2020 15:16 
# @Author : anlei
# @File : 线性回归.py

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def myLinear():
    """
    线性回归直接预测房子价格
    :return: None
    """

    # 获取数据
    lb = load_boston()
    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # print(y_train,y_test)
    # 进行标准化处理 (这里对特征值处理，所有数据都变小了，所有对应的目标值也应该变小了)
    # 特征值和目标值都需要进行标准化处理，要实例化两个标准化API
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值标准化
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))     # 0.19版本之后要求传入数据必须是二维
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 区别y_train 和y_train.ravel()

    # print(y_train)   # 二维数组
    #
    # print(y_train.ravel())   # 把多维数组转换成以为数组

    # estimator预测

    # 正规方程求解方式预测结果

    lr = LinearRegression()

    lr.fit(x_train, y_train.ravel())

    print(lr.coef_)

    # 预测测试集的房子价格
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))

    # 返回标准化之前的数据

    print("测试集里面每个房子的预测价格：\n", y_lr_predict)

    print("正规方程的均方误差", mean_squared_error((std_y.inverse_transform(y_test)), y_lr_predict))

    # 梯度下降求解方式预测结果

    sgd = SGDRegressor()

    sgd.fit(x_train, y_train.ravel())

    print(sgd.coef_)

    # 预测测试集的房子价格
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))

    # 返回标准化之前的数据

    print("测试集里面每个房子的预测价格：\n", y_sgd_predict)

    print("梯度下降的均方误差", mean_squared_error((std_y.inverse_transform(y_test)), y_sgd_predict))


    # 岭回归
    rd = Ridge(alpha=1.0)

    rd.fit(x_train,y_train.ravel())

    print(rd.coef_)

    # 预测测试集的房子价格
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))

    # 返回标准化之前的数据

    print("测试集里面每个房子的预测价格：\n", y_rd_predict)

    print("梯度下降的均方误差", mean_squared_error((std_y.inverse_transform(y_test)), y_rd_predict))

    return None


if __name__ == "__main__":
    myLinear()