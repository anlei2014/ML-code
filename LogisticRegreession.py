# -*- coding: utf-8 -*- 
# @Time : 18/11/2020 20:41 
# @Author : anlei
# @File : LogisticRegreession.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 构造列标签名字
column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# 读取数据
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column)

print(data)

# 缺失值进行处理
data = data.replace(to_replace='?', value=np.nan)

data = data.dropna()

# 进行数据的分割
x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]],data[column[10]],test_size=0.25)

# 进行标准化处理
std = StandardScaler()

x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# 逻辑回归
lg = LogisticRegression(C=1.0)
lg.fit(x_train,y_train)

print(lg.coef_)

y_predict = lg.predict(x_test)

print("准确率：",lg.score(x_test,y_test))

print("召回率：",classification_report(y_test, y_predict, labels=[2,4], target_names=["良性","恶性"]))