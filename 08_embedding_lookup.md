



## (1)符号嵌入

`tf.nn.embedding_lookup (params,ids,partition_strategy=’mod’,validate_indices=True)`

+ 根据索引ids查询embedding列表params中的tensor值。
+ 如果len(params) > 1，id将会安照partition_strategy策略进行分割 
  + 如果partition_strategy为”mod”， id所分配到的位置为p = id % len(params) 比如有13个ids，分为5个位置，那么分配方案为： [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]] 
  + 如果partition_strategy为”div”,那么分配方案为： [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]

首先来看len(params)==1的情况

```PYTHON
x = tf.constant(np.arange(20).reshape(5, 4))
a = tf.nn.embedding_lookup(x, ids=[0, 3, 4, 2, 1])
b = tf.nn.embedding_lookup(x, ids=[[0, 1], [2, 3], [4, 4]])
----------------------------------------------
a的结果：
[[ 0  1  2  3]
 [12 13 14 15]
 [16 17 18 19]
 [ 8  9 10 11]
 [ 4  5  6  7]]
b的结果：
[[[ 0  1  2  3]
  [ 4  5  6  7]]

 [[ 8  9 10 11]
  [12 13 14 15]]

 [[16 17 18 19]
  [16 17 18 19]]]
```

在看len(params)==1的情况

```PYTHON
x = tf.constant(np.arange(20).reshape(5, 4))
y = tf.constant(np.arange(20, 60).reshape(-1, 4))
c = tf.nn.embedding_lookup([x, y, x],
                           partition_strategy='mod', ids=[1, 0, 8, 2])
----------------------------------------------------------
[[20 21 22 23]
 [ 0  1  2  3]
 [ 8  9 10 11]
 [ 0  1  2  3]]
```

这里x.shape=(5,4)且y.shape=(10,4)，共有(5+10+5)=20个id，所以的取值范围是ids=[0,1,2,...,15)，由于strategy取值是'mod'即需要划分为3份（这里len([x,y,x])=3）,即(0,3,6,9,12,15,18) (1,4,7,10,13,16,19)和(2,5,8,11,14,17)。

对ids=[1,0,8,2]中的1划分在了第二份数据y中，在y中索引是第1个即[20 21 22 23] ；

对ids=[1,0,8,2]中的0划分在了第三份数据x中，在x中索引是第1个[ 0  1  2  3]；

 对ids=[1,0,8,2]中的8划分在了第三份数据y中，在y中索引是第3个[ 8  9 10 11]；

对ids=[1,0,8,2]中的2划分在了第三份数据x中，在x中索引是第1个[ 0  1  2  3]；

在看另外一个例子：

```PYTHON
x = tf.constant(np.arange(20).reshape(5, 4))
y = tf.constant(np.arange(20, 60).reshape(-1, 4))
c = tf.nn.embedding_lookup([x, y, x],
                           partition_strategy='div', ids=[1, 0, 8, 2])
----------------------------------------------------------
[[ 4  5  6  7]
 [ 0  1  2  3]
 [24 25 26 27]
 [ 8  9 10 11]]
```

这里x.shape=(5,4)且y.shape=(10,4)，所以的取值范围是ids=[0,1,2,...,15)，由于strategy取值是'div'即需要划分为3份（这里len([x,y,x])=3）,即(0,1,2,3,4,5,6) (7,8,9,10,11,12,13)和(14,15,16,17,18,19)。

对ids=[1,0,8,2]中的1划分在了第一份数据x中，在x中索引是第2个即[ 4  5  6  7] ；

对ids=[1,0,8,2]中的0划分在了第一份数据x中，在x中索引是第1个[ 0  1  2  3]；

 对ids=[1,0,8,2]中的8划分在了第二份数据y中，在y中索引是第2个[24 25 26 27]；

对ids=[1,0,8,2]中的2划分在了第一份数据x中，在x中索引是第3个 [ 8  9 10 11]；

