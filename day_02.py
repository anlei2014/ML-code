# -*- coding: utf-8 -*- 
# @Time : 15/11/2020 21:12 
# @Author : anlei
# @File : day_02.py

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split
# 实例化
li = load_iris()

# # 获取特征值
# print("特征值")
# print(li.data)
#
# # 获取目标值
# print("目标值")
# print(li.target)
#
# # 描述
# print("描述")
# print(li.DESCR)
#
# print("特征名")
# print(li.feature_names)

# # 注意返回值，训练集train x_train  y_train  测试集test  x_test ,y_test
# x_train,x_test,y_train,y_test = train_test_split(li.data, li.target, test_size=0.25)
#
# print("训练集特征值和目标值",x_train,y_train)
# print("测试集特征值和目标值",x_test,y_test)

# news = fetch_20newsgroups(subset="all")
#
# print(news.data)
# print(news.target)

