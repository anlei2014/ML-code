# -*- coding: utf-8 -*- 
# @Time : 14/11/2020 15:12 
# @Author : anlei
# @File : 中文文本处理.py


from sklearn.feature_extraction.text import CountVectorizer
import jieba

def cutwords():
    """
    对中文文本进行处理
    :return: C1,C2,C3
    """
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃明天。")

    con2 = jieba.cut("我们看到的从很远信息来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物整整含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成list
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 转成字符串

    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)



    return c1, c2, c3


def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutwords()

    print(c1,c2,c3)

    cv = CountVectorizer()

    data = cv.fit_transform([c1,c2,c3])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


if __name__ == "__main__":
    hanzivec()