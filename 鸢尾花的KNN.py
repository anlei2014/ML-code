# -*- coding: utf-8 -*- 
# @Time : 17/11/2020 11:09 
# @Author : anlei
# @File : 鸢尾花的KNN.py

from matplotlib import pyplot as plt
# from matplotlib import patches as mpatches
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# 实例化
iris = load_iris()
# 打印   四个数值分别为萼片和花斑的数据（长和宽）
# print(iris.data)

# 花卉的种类 三种取值  0，1，2
special = iris.target
# print(special)

# 每个取值代表的花卉名称
# print(iris.target_names)

# 二维数值的第一列
# x  = iris.data[:,0]
# # print(x)
#
# # 二维数组的的第二列
# y = iris.data[:,1]
#
# # 设置标题
# plt.figure()
# plt.title('length and width')
# plt.xlabel('length')
# plt.ylabel('width')
# plt.scatter(x,y,c=special)
# plt.show()

pca = PCA(n_components=3)
x_reduce = pca.fit_transform(iris.data)

fig = plt.figure()
ax = Axes3D(fig)

# 设置标题
ax.set_title("iris dataset by pca")
ax.scatter(x_reduce[:,0],x_reduce[:,1],x_reduce[:,2],c=special)

plt.savefig('fig.png', bbox_inches='tight')
