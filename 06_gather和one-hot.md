## (1)tf.gather的用法

`gather(params, indices, validate_indices=None, name=None, axis=0)`

可以把向量中的某些索引值取出来，得到新的向量，适用于提取的索引为不连续的情况。

```PYTHON
a = tf.constant([[1, 2, 3, 4], [4, 5, 6, 7]])

b = tf.gather(a, [1, 0], axis=0)
c = tf.gather(a, [2, 3, 0], axis=1)
--------------------------------------
[[4 5 6 7]
 [1 2 3 4]]

[[3 4 1]
 [6 7 4]]
```



## (2)tf.one-hot的用法

`one_hot(indices, depth, on_value=None,off_value=None,axis=None,dtype=None,name=None)`

将数据进行one-hot编码，参数的含义

indices是需要编码的数字。depth表示one-hot的编码的长度。如果出现该编码则赋值on_value，默认值为1，不出现则编码off_value，默认值为0。按照axis轴进行 编码，默认值axis=1

```
a = tf.constant([2, 1, 5, 1, 1])

b = tf.one_hot(a, depth=3)
c = tf.one_hot(a, depth=3, axis=0)
d = tf.one_hot(a, depth=3, axis=1)
------------------------------------------
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]]
 
[[0. 0. 0. 0. 0.]
 [0. 1. 0. 1. 1.]
 [1. 0. 0. 0. 0.]]

[[0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]]
```





























