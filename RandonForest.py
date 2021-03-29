# -*- coding: utf-8 -*- 
# @Time : 17/11/2020 20:42 
# @Author : anlei
# @File : decisiontree.py

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
# 随机森林进行预测（超参数调优）
rf = RandomForestClassifier()

param = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5,8,15,25,30]}

# 网格搜索与交叉验证
gc = GridSearchCV(rf,param_grid=param,cv=2)

gc.fit(x_train,y_train)

print("准确率",gc.score(x_test,y_test))

print("查看选择的参数模型：",gc.best_params_)




