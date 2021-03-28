# -*- coding: utf-8 -*- 
# @Time : 14/11/2020 11:57 
# @Author : anlei
# @File : day_01.py

from sklearn.feature_extraction import DictVectorizer

from sklearn.feature_extraction.text import CountVectorizer


def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform
    data = dict.fit_transform([{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}])

    # 字典数据抽取，把字典中一些类别数据，分别进行转换成特征

    # 数组形式，有类别的这些特征先转换字典数据
    print(dict.get_feature_names())

    # 转回之前的数据
    print(dict.inverse_transform(data))

    print(data)

    return None

def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    # 实例化CountVectorizer
    vector = CountVectorizer()

    # 调用fit_transform输入并转换数据

    # res = vector.fit_transform(["life is short,i like python","life is too long, i dislike python"])
    res = vector.fit_transform(["人生苦短，我喜欢python","人生漫长，不用python"])

    # 打印结果
    print(vector.get_feature_names())

    # 对返回的结果用toarray()进行转换
    print(res.toarray())

    return None


if __name__ == "__main__":

    dictvec()
    countvec()