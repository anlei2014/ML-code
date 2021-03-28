# -*- coding: utf-8 -*- 
# @Time : 14/11/2020 19:22 
# @Author : anlei
# @File : 缺失值.py


import numpy as np
from sklearn.impute import SimpleImputer


def im():
    """

    :return:
    """
    im = SimpleImputer(missing_values = np.nan,strategy='mean')
    data = im.fit_transform([[1,2],[np.nan,3],[7,6]])

    print(data)

    return None


if __name__ == "__main__":
    im()
