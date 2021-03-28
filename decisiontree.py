# -*- coding: utf-8 -*- 
# @Time : 17/11/2020 20:42 
# @Author : anlei
# @File : decisiontree.py

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
# 获取数据
titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

# 处理数据，找出特征值和目标值
x = titan[['pclass','age','sex']]

y = titan['survived']

print(x)

# 处理缺失值
x['age'].fillna(x['age'].mean(), inplace=True)

# 分割数据集到训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

# 进行处理（特征工程）  特征-》类别-》one_hot编码
dict = DictVectorizer(sparse=False)

x_train = dict.fit_transform(x_train.to_dict(orient="records"))

print(dict.get_feature_names())

x_test = dict.transform(x_test.to_dict(orient="records"))

# print(x_train)
# 用决策树进行预测
dec = DecisionTreeClassifier(max_depth=10)

dec.fit(x_train,y_train)

y_predict = dec.predict(x_test)

print(y_predict)

print(dec.score(x_test,y_test))

# 导出决策树的结构
# export_graphviz(dec,out_file="./tree.dot",feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])



