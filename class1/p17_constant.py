import tensorflow as tf

# a = tf.constant([1, 5], dtype=tf.int64)
a = tf.constant([1, 5])
print("a:", a)
print("a.dtype:", a.dtype)
print("a.shape:", a.shape)

# shape=(2,) 逗号之前有几个数组代表几维，数字的值代表的数据个数

# 本机默认 tf.int32  可去掉dtype试一下 查看默认值
