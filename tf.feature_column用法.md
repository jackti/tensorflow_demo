# tf.feature_column用法

Tensorflow平台提供了FeatureColumn API为特征工程提供了强大的支持。FeatureColumn是原始数据和Estimator模型之间的桥梁，被用来把各种形式的原始数据转换为模型能够使用的格式。

神经网络只能处理数值数据，网络中的每个神经元节点执行一些针对输入数据和网络权重的乘法和加法运算。然而，在现实中有很多的非数值类别数据，如产品的品牌、类目等等，这些数据必须进行装换否则神经网络是无法处理的。另一方面，即使是数值数据，在扔给神经网络训练之前也需要进行一些处理，如标准化、离散化等等。



在Tensorflow中，通过`tf.feature_column`模块来创建feature columns。有两大类feature columns，一类是生成dense tensor的Dense Column；另一类是生成sparse tensor的Categorical Column。



### Numeric Column

```python
tf.feature_column.numeric_column(
    key,
    shape=(1,),
    default_value=None,
    dtype=tf.float32,
    normalizer_fn=None
)
```

+ key：特征的名字，即对应的列名称，必须是字符串。
+ shape：该key所对应特征的shape，默认是1。
+ default_value：如果不存在使用的默认值。
+ normalizer_fn：该特征下所有数据进行装换的函数，这里不仅限与normalize，也可以是任何的转换方法。

```PYTHON
def test_numeric():
    features = {'f1': [[1.0], [3.0], [2.0]]}
    builder = _LazyBuilder(features)

    column = tf.feature_column.numeric_column('f1')
    f1_column = tf.feature_column.input_layer(features, [column])

    with tf.Session() as sess:
       sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        print(builder)
        print(sess.run(f1_column))
        
test_numeric() 
```



### Bucketized Column

BucketizedColumn用来将Numeric Column的值按照提供的边界离散化为多个值。

```PYTHON
tf.feature_column.bucketized_column(
    source_column,
    boundaries
)
```

+ source_column：必须是numeric_column
+ boundaries：分桶边界。如boundaries=[0.,1.,2.]产生的分桶就是`(-inf,0) [0,1) [1,2)`和`[2,+inf)` 相当于分了4个桶

```PYTHON
def test_bucket():
    features = {'f1': [[1.0], [3.0], [2.0]]}
    builder = _LazyBuilder(features)

    f1_num_column = tf.feature_column.numeric_column('f1')
    f1_bucket_column = tf.feature_column.bucketized_column(f1_num_column, boundaries=[1.5, 2.5])
    f1_column = tf.feature_column.input_layer(features, [f1_bucket_column])

    with tf.Session() as sess:
       sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        print(builder)
        print(sess.run(f1_column))
```



### Categorical Identity Column

主要处理连续的数字类。与Bucketized Column类似，但是CategoryIdentityColumn使用单一值表示bucket。比如ID一共有10000个，那么可以使用该方式。但是如果多数ID都没有被使用，那么还是建议使用`category_column_with_hash_bucket`。

```PYTHON
def test_category_identity():
    features = {'f1': [[1], [3], [2], [9]]}
    builder = _LazyBuilder(features)

    f1_identity_column = tf.feature_column.categorical_column_with_identity('f1', num_buckets=5, default_value=0)
    f1_indicator_column = tf.feature_column.indicator_column(f1_identity_column)
    f1_column = tf.feature_column.input_layer(features, [f1_indicator_column])

    with tf.Session() as sess:
     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
     print(builder)
     print(sess.run(f1_column))
```

`categorical_column_with_identity`返回的是sparse tensor，但是`tf.feature_column.input_layer`只接收dense tensor。所以在这里使用indicator_column将稀疏tensor转化为one-hot或者multi-hot形式的稠密tensor。





### Categorical Vocabulary Column

把一个vocabulary中的string或int映射为数值型的类别特征，可以方便的进行one-hot编码。在Tensorflow中有两种提供词汇表的方法，一种是使用list，另一种是使用file：

```python
tf.feature_column.categorical_column_with_vocabulary_list(
    key,
    vocabulary_list,
    dtype=None,
    default_value=1,
    num_oov_buckets=0
)
```

```python
tf.feature_column.categorical_column_with_vocabulary_file(
    key,
    vocabulary_file,
    dtype=None,
    default_value=1,
    num_oov_buckets=0
)
```

+ key：特征名称
+ dtype：仅仅支持string和int，其他类型数据不支持该操作
+ default_value：当不在vocabulary中的默认值，此时num_oov_buckets必须是0
+ num_oov_buckets：用来处理不在vocabulary中的值，如果是0则使用default_value进行填充；如果大于0，则会在`[len(vocabulary),len(vovabulary)+num_oov_buckets]`这个区间上重新计算当前特征的值。

两个函数的唯一区别在vocabulary存储的位置，vocabulary_list表示存储在list中，vocabulary_file表示词汇表的文件名。

```PYTHON
def test_category_vocabulary_list():
    features = {'f1': [[1], [3], [2], [9]]}
    builder = _LazyBuilder(features)

    f1_category_column = tf.feature_column.categorical_column_with_vocabulary_list('f1', vocabulary_list=[1, 2, 3])
    f1_indicator_column = tf.feature_column.indicator_column(f1_category_column)
    f1_column = tf.feature_column.input_layer(features, [f1_indicator_column])

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run(tf.tables_initializer())

        print(builder)
        print(sess.run(f1_column))
```



### Categorical Hash Column

当category的数量很多，无法使用指定category的方法来处理了。那么，可以使用Hash的方法来进行处理。如，文本切词以后，每个词都可以使用这种方式处理（当然使用categorical_column_with_vocabulary_file也是一种不错的选择，将高频词选择出来保存在文件中）。

```PYTHON
def test_category_hash_bucket():
    features = {'f1': [['B'], ['G'], ['R'], ['A']]}
    builder = _LazyBuilder(features)

    f1_hash_column = tf.feature_column.categorical_column_with_hash_bucket('f1', hash_bucket_size=7)
    f1_indicator_column = tf.feature_column.indicator_column(f1_hash_column)
    f1_column = tf.feature_column.input_layer(features, [f1_indicator_column])

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run(tf.tables_initializer())

        print(builder)
        print(sess.run(f1_column))
```



### Crossed Column

交叉特征是一种很常用的特征工程手段，尤其是在使用LR模型时，CrossedColumn仅仅适用于Sparser特征，产生的依然是Sparsor特征。

```PYTHON
tf.feature_column.crossed_column(
    keys,
    hash_bucket_size,
    hash_key=None
)
```

```python
def test_crossed_column():
    featrues = {
        'price': [['A'], ['B'], ['C']],
        'color': [['R'], ['G'], ['B']]
    }
    price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C', 'D'])
    color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
    p_x_c = feature_column.crossed_column([price, color], 16)
    p_x_c_identy = feature_column.indicator_column(p_x_c)
    p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([p_x_c_identy_dense_tensor]))
test_crossed_column()
```



### Embedding Column

**embedding column和Indicator column都不能直接作用在原始特征上**，而是作用在categorical columns上的。上面已经使用过了indicator column将categorical column得到的稀疏tensor转换为one-hot或multi-hot形式的稠密tensor。

当某些特征的类别数量非常大时，使用indicator column把原始数据转换为神经网络的输入就非常不灵活，这时通常是使用embedding column把原始特征映射为一个低维稠密的实数向量。同一类别的embedding向量间的距离通常可以用来度量类别直接的相似性。

```PYTHON
tf.feature_column.embedding_column(
    categorical_column,
    dimension,
    combiner='mean',
    initializer=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True
)
```

+ categorical_column：使用categorical_column产生的sparser column
+ dimension：定义embedding的维度
+ combiner：对于多个entries进行推导，默认是mean
+ initializer：初始化方法，默认使用高斯分布来初始化
+ tensor_name_in_ckpt：是否可以从check point中恢复
+ ckpt_to_load_from：指定恢复的文件，在tensor_name_in_ckpt不为空的情况下设置
+ max_norm：默认是l2
+ trainable：是否可训练，默认是true

```pytho
def test_embedding():
    tf.set_random_seed(1)
    color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_column_tensor = color_column._get_sparse_tensors(builder)

    color_embeding = feature_column.embedding_column(color_column, 4, combiner='sum')
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print('embeding' + '_' * 40)
        print(session.run([color_embeding_dense_tensor]))

test_embedding()
```

输出结果是：

```
[SparseTensorValue(indices=array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1],
       [2, 0],
       [2, 1],
       [3, 0],
       [3, 1]], dtype=int64), values=array([ 0,  1,  1, -1,  2,  2, -1, -1], dtype=int64), dense_shape=array([4, 2], dtype=int64))]
embeding________________________________________
[array([[-0.8339818 , -0.4975947 ,  0.09368954,  0.16094571],
       [-0.6342659 , -0.19216162,  0.18877633,  0.17648602],
       [ 1.5531666 ,  0.27847385,  0.12863553,  1.2628161 ],
       [ 0.        ,  0.        ,  0.        ,  0.        ]],
      dtype=float32)]
```

从上面的结果可以看出不在vocabulary里的数据'A'在经过`categorical_column_with_vocabulary_list`操作时映射为默认的-1，而默认值-1在embedding column时映射为0向量，这个特性很有用——可以使用-1来填充一个不定长的ID序列，这样就可以得到定长的序列，然后经过embedding column之后，填充的-1值不影响原来的结果。

有时候在同一个网络模型中，很多特征可能需要共享相同的embedding映射空间，比如用户历史行为序列中商品ID和候选商品ID，这时可以使用`tf.feature_column.shared_embedding_columns`

```python 
tf.feature_column.shared_embedding_columns(
    categorical_columns,
    dimension,
    combiner='mean',
    initializer=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True
)
```

+ categorical_columns为需要共享的embedding映射空间的类别特征列表
+ 其他参数和embedding column类似

```PYTHON
def test_shared_embedding_column_with_hash_bucket():
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run(color_dense_tensor))

test_shared_embedding_column_with_hash_bucket()



-------------------------------------------------------------
not use input_layer________________________________________
[SparseTensorValue(indices=array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1],
       [2, 0],
       [3, 0],
       [3, 1]], dtype=int64), values=array([5, 5, 1, 1, 2, 2, 2], dtype=int64), dense_shape=array([4, 2], dtype=int64))]
[SparseTensorValue(indices=array([[0, 0],
       [1, 0],
       [3, 0]], dtype=int64), values=array([5, 1, 2], dtype=int64), dense_shape=array([4, 1], dtype=int64))]
<class 'list'>
use input_layer________________________________________
[[ 0.37802923 -0.27973637  0.11547407  0.75605845 -0.55947274  0.23094814]
 [-0.5264772   0.86587846 -0.36023238 -1.0529544   1.7317569  -0.72046477]
 [ 0.          0.          0.         -0.9269535  -0.17690836  0.42011076]
 [-0.9269535  -0.17690836  0.42011076 -1.853907   -0.35381672  0.8402215 ]]
```

`tf.feature_column.shared_embedding_columns`的返回值是一个与参数categorical_columns维数相同的列表。



### Weighted Categorical Column

有时需要给一个类别特征赋予一定的权重，如给用户行为序列按照行为发生的时间到某个特定的时间的差来计算不同的权重，可以使用`weighted_categorical_column`

```python
tf.feature_column.weighted_categorical_column(
    categorical_column,
    weight_feature_key,
    dtype=tf.float32
)
```

```python
def test_weighted_categorical_column():
    color_data = {'color': [['R'], ['G'], ['B'], ['A']],
                  'weight': [[1.0], [2.0], [4.0], [8.0]]}  # 4行样本
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_weight_categorical_column = feature_column.weighted_categorical_column(color_column, 'weight')
    builder = _LazyBuilder(color_data)
    with tf.Session() as session:
        id_tensor, weight = color_weight_categorical_column._get_sparse_tensors(builder)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('weighted categorical' + '-' * 40)
        print(session.run([id_tensor]))
        print('-' * 40)
        print(session.run([weight]))
test_weighted_categorical_column()
---------------------------------------
weighted categorical----------------------------------------
[SparseTensorValue(indices=array([[0, 0],
       [1, 0],
       [2, 0],
       [3, 0]], dtype=int64), values=array([ 0,  1,  2, -1], dtype=int64), dense_shape=array([4, 1], dtype=int64))]
----------------------------------------
[SparseTensorValue(indices=array([[0, 0],
       [1, 0],
       [2, 0],
       [3, 0]], dtype=int64), values=array([1., 2., 4., 8.], dtype=float32), dense_shape=array([4, 1], dtype=int64))]
```

可以看到，相对于前面其他categorical_column来说多了weight这个tensor。weighted_categorical_column的一个用例就是，weighted_categorical_column的结果传入给shared_embedding_columns可以对ID序列的embeding向量做加权融合。

 















