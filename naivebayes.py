# -*- coding: utf-8 -*- 
# @Time : 17/11/2020 14:30 
# @Author : anlei
# @File : naivebayes.py

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

news = fetch_20newsgroups(subset="all")

# 对数据就行分割
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)

# 对数据集就行特征抽取
tf = TfidfVectorizer()

# 以训练集当中的词的列表进行每篇文章重要性的统计
x_train = tf.fit_transform(x_train)

# print(tf.get_feature_names())

x_test = tf.transform(x_test)

# 进行算法的预测
mlt = MultinomialNB(alpha=1.0)

print(x_train)

mlt.fit(x_train,y_train)

y_predict = mlt.predict(x_test)

print(y_predict)

# 得出准确率为
print(mlt.score(x_test,y_test))


