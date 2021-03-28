# -*- coding: utf-8 -*- 
# @Time : 14/11/2020 18:59 
# @Author : anlei
# @File : 标准化.py

from sklearn.preprocessing import StandardScaler


def stand():
    """
    标准化缩放
    :return: None
    """
    std = StandardScaler()
    data = std.fit_transform([[1,-1,3],[2,4,2],[4,6,-1]])

    print(data)

    return None


if __name__ == "__main__":
    stand()