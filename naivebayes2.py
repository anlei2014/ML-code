# -*- coding: utf-8 -*- 
# @Time : 17/11/2020 15:00 
# @Author : anlei
# @File : naivebayes2.py


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


news = fetch_20newsgroups(subset="all")

x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25)

tf = TfidfVectorizer()
x_train = tf.fit_transform(x_train)

x_test = tf.transform(x_test)

mlt = MultinomialNB(alpha=1.0)

mlt.fit(x_train,y_train)

y_predict = mlt.predict(x_test)

score = mlt.score(x_test,y_test)
print(score)
