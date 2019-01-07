## (1) tf.unstack的用法

`unstack(value,num=None,axis=0,name='unstack' )`

将秩为R的张量分解为秩为R-1的张量。其参数含义：

value:输入的张量（其实就是一个多维数组）

num:默认值是None，表示需要分解的个数，若为None则均分

axis:指明需要对哪个维度进行分解，默认值是axis=0

```python
a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.unstack(a, num=2, axis=0)
c = tf.unstack(a, num=3, axis=1)
d = tf.unstack(a, axis=1)
-------------------------------->
[array([1, 2, 3], dtype=int32), array([4, 5, 6], dtype=int32)]
[array([1, 4], dtype=int32), array([2, 5], dtype=int32), array([3, 6], dtype=int32)]
[array([1, 4], dtype=int32), array([2, 5], dtype=int32), array([3, 6], dtype=int32)]
```



## (2) tf.slice的用法

`slice(input_, begin, size, name=None)`

从输入数据input_中提取出一块切片。其参数含义为：

input_：表示输入的张量

begin：切片的起始位置

size：表示输出tensor的数据维度，其中size[i]表示在第i维度上的元素个数

```PYTHON
input = tf.constant([[[1, 1, 1], [2, 2, 2]],  
                     [[3, 3, 3], [4, 4, 4]],  
                     [[5, 5, 5], [6, 6, 6]]]) 

a = tf.slice(input, begin=[1, 0, 0], size=[1, 1, 3])                                 b = tf.slice(input, begin=[1, 0, 0], size=[2, 1, 3])  
c = tf.slice(input, begin=[1, 0, 0], size=[2, 2, 2])  
"""[1,0,0]表示第一维偏移了1
则是从[[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]]中选取数据
然后选取第一维的第一个，第二维的第一个数据，第三维的三个数据"""

[[[3 3 3]]

 [[5 5 5]]]
--------------------
[[[3 3]
  [4 4]]
-------------------
 [[5 5]
  [6 6]]]
```



## tf.split的用法

`tf.split( value, num_or_size_splits, axis=0, num=None, name='split' )`

用来切割张量，value表示需要切割的张量，该函数有两种使用方法，对输入的tensor为`20*30*40`的张量

+ 如果`num_or_size_splits`传入是一个整数，这个整数代表输入张量最后被切分成几个小张量，axis的数值代表切割哪个维度。若调用`split(tensor,2,0)`返回两个`10*30*40`的小张量；
+ 如果`num_or_size_splits`传入的是一个向量，这个向量有几个分量就分成几份，切割维度有axis决定。若调用`split(tensor,[10,5,25],2)`则返回三个分量为`20*30*10`、`20*30*5`和`20*30*25`。输入的`num_or_size_splits`各个分量加和必须等于axis所指示原张量的大小。即10+5+25=40。

```PYTHON
a = tf.constant([[1, 2, 3], [4, 5, 6]])

b = tf.split(a, num_or_size_splits=2, axis=0)
c = tf.split(a, num_or_size_splits=3, axis=1)
d = tf.split(a, num_or_size_splits=[1, 2], axis=1)
--------------------------------------------------------------------
[array([[1, 2, 3]], dtype=int32), array([[4, 5, 6]], dtype=int32)]

[array([[1],
       [4]], dtype=int32), 
 array([[2],
       [5]], dtype=int32), 
 array([[3],
       [6]], dtype=int32)]

[array([[1],
       [4]], dtype=int32), 
 array([[2, 3],
       [5, 6]], dtype=int32)]
```



















