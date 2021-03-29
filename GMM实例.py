# -*- coding: utf-8 -*- 
# @Time : 20/11/2020 20:29 
# @Author : anlei
# @File : GMM实例.py

import pandas as pd
import matplotlib.pyplot as plt

file_path = "./fremont.csv"
# data1 = pd.read_csv(file_path, index_col='Date')
data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
# 删除列
data = data.drop(["Fremont Bridge Total"], axis=1)
# print(data1.head())
print(data.head())



