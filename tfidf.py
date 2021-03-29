# -*- coding: utf-8 -*- 
# @Time : 14/11/2020 16:42 
# @Author : anlei
# @File : tfidf.py

from 中文文本处理 import cutwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def tfidfvec():
    """

    :return:
    """

    c1,c2,c3 = cutwords()

    print(c1,c2,c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1,c2,c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


if __name__ == "__main__":
    tfidfvec()