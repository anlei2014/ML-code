# -*- coding: utf-8 -*- 
# @Time : 15/11/2020 12:58 
# @Author : anlei
# @File : 数据降维之Filter.py


from sklearn.feature_selection import VarianceThreshold


def vt():

    """
    特征选择-删除低方差的特征
    :return: None
    """
    vt = VarianceThreshold()    #()参数可以自己写threshold = 0.0/1.0

    data = vt.fit_transform([[0,2,0,3],[0,1,4,3],[0,1,1,3]])

    print(data)

    return None


if __name__ == "__main__":
    vt()