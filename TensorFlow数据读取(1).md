# TensorFlow数据读取

TensorFlow程序读取数据的方式常用的有：

> **(1)预加载数据(Preload)**：在TensorFlow图中定义常量或者变量来保存数据；
>
> **(2)供给数据(Feeding)**：利用`placehold`和`feed_dict`的方式，在设计Graph的时候保留占位符，当程序运行时向占位符中传递数据，"喂给"后端训练；
>
> **(3)文件读取数据**：基于队列的方式读取数据，可以一边使用数据一边从磁盘读取数据。



## 一、预加载数据Preload

Preload有两种方法：①存储在常数constant中；②存储在变量variable中,初始化或后续均可更新。这种读取数据的方式只适合小数据，通常在程序中定义的固定值，如循环次数等等，很少直接用来读取训练数据。

```PYTHON
x1 = tf.constant([2, 3, 1])
x2 = tf.constant([1, 1, 3])
y = tf.add(x1, x2)
with tf.Session() as sess:
    print(sess.run(y))
```



## 二、Feeding方式

TensorFlow的数据供给机制可以将数据"喂给"运算图中任一张量中，通过给`run()`或者`eval()`函数输入`feed_dict`参数可以启动运算过程。

```PYTHON
# 设计Graph
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.add(x1, x2)
# 产生数据
l1 = [2, 3, 1]
l2 = [1, 1, 3]
# 在session中供给数据，计算y
with tf.Session() as sess:
    print(sess.run(y, feed_dict={x1: l1, x2: l2}))
```



## 三、文件读取数据

上面的两种方式适用于数据量较小的情况，当运行大数据就会十分消耗内存。因此，最好的方法是使用生产者消费者模式，让一个进程让TF从文件中读取数据，而另一个进程进行异步计算，防止等待硬盘IO。

### 读取文件格式举例

TensoFlow可读取的文件类型包括了CSV文件，二进制文件、TFRecords文件和图像文件等。根据读取文件的不同格式，需要选择不同的**文件阅读器**，然后将文件名队列提供给**阅读器`read`**方法，阅读器会输出一个key来表征输入的文件和其中的记录（可以方便的调试）,同时得到一个**字符串标量**，这个字符串标量可以被一个或多个**解析器**解析，或者转换操作将其解码为张量构成样本数据。

#### csv文件

+ 阅读器 `tf.TextLineReader`
+ 解析器`tf.decode_csv`
+ 例子

```PYTHON
filename_queue = tf.train.string_input_producer(['./data/1a.csv', './data/2b.csv'], num_epochs=2, shuffle=False)
read = tf.TextLineReader()
_, value = read.read(filename_queue)

record_default = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, label = tf.decode_csv(value, record_defaults=record_default, field_delim=",")

feature = tf.stack([col1, col2, col3, col4])
batch_feature, batch_label = tf.train.batch([feature, label], batch_size=6)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        while not coord.should_stop():
            print(sess.run([batch_feature, batch_label]))
    except tf.errors.OutOfRangeError:
        print("All data done!")
    finally:
        coord.request_stop()

    coord.join(threads)
```

其中涉及到几个重要的函数(1)`tf.train.string_input_producer(string_tensor,num_epochs=None,shuffle=True,capacity=32)`

参数：`string_tensor`是读入文件列表；`num_epochs`如果指定的话，每个文件数据会重复num_epochs次数；反之，若不设定，那么每个文件数据会被无限制的生成；`shuffle`输入的文件是否随机打乱。

(2)`tf.decode_csv(records, record_defaults, field_delim=None, name=None)`

参数：`records` 为reader读到的内容，为CSV(TXT)的一行。`field_delim`默认为逗号，指定一行里面的值使用逗号或者空格等其他符号隔开。`record_defaults` 用于指定矩阵格式以及数据类型，[1]里面的1指定数据类型，为整数，若为float，则应该变为[1.0]，若是字符串则为['null']

调用`run`或者`eval`去执行`read`之前， 必须先调用`tf.train.start_queue_runners`来将文件名填充到队列。否则`read`操作会被阻塞到文件名队列中有值为止。

#### 二进制文件

+ 阅读器：`tf.FixedLengthRecordReader`
+ 解析器：`tf.decode_raw`



#### 图像文件

+ 阅读器：`tf.WholeFileReader`
+ 解析器：`tf.image.decode_image`,`tf.image.decode_jpeg`,`tf.image.decode_bmp`...
+ 例子

```PYTHON
import tensorflow as tf

filename_queue = tf.train.string_input_producer(['../data/1.png', '../data/2.png'], num_epochs=2)
reader = tf.WholeFileReader()
_, value = reader.read(filename_queue)
image_tensor = tf.image.decode_png(value)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    try:
        while not coord.should_stop():
            print(sess.run(image_tensor))
    except tf.errors.OutOfRangeError:
        print("All Data Done!")
    finally:
        coord.request_stop()

    coord.join(threads)
```



#### TFRecords文件

TFRecords文件是TensorFlow的标准格式，使用TFRecords文件保存记录的方法可以将任意数据转换为TensorFlow所支持的格式，使得数据集与网络应用架构更匹配。

将原始数据转换为TFRecords格式，其主要步骤可以分为两步：

> (1)获取数据，将数据填入到`tf.train.Example`协议内存块(protocol buffer)，将协议内存块序列化为一个字符串
>
> (2)通过`tf.python_io.TFRecordWriter`写入到TFRecords文件

```PYTHON
def toList(value):
    if type(value)==list:
        return value
    else: 
        return [value]

def _int64_feature(value):
    value = toList(value)
    value = [int(x) for x in value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    value = toList(value)
    value = [float(x) for x in value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
def _bytes_feature(value):
    value = toList(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def convert_to(data_set, name):
        images = data_set.images
        labels = data_set.labels
        num_examples = data_set.num_examples

        if images.shape[0] != num_examples:
            raise ValueError('Images size %d does not match label size %d.' %
                        (images.shape[0], num_examples))
        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        filename = os.path.join(FLAGS.directory, name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
                image_raw = images[index].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
                writer.write(example.SerializeToString())
         writer.close()
```

从TFRecords文件中读取数据：

+ 阅读器：`tf.TFRecordReader`
+ 解析器：`tf.parse_single_example` 或者`tf.parse_example`
+ 例子

**(1) tf.parse_single_example()读取数据 **

```PYTHON
# 第一步：建立文件名队列
filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=3)

def read_single(filename_queue, shuffle_batch, if_enq_many):
    # 第二步： 建立阅读器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 第三步：根据写入时的格式建立相对应的读取features
    features = {
        'sample': tf.FixedLenFeature([5], tf.int64),# 如果不是标量，要说明数组的长度
        'label': tf.FixedLenFeature([], tf.int64)
    }
    # 第四步： 用tf.parse_single_example()解析单个EXAMPLE PROTO
    Features = tf.parse_single_example(serialized_example, features)
    # 第五步：对数据进行后处理
    sample = tf.cast(Features['sample'], tf.float32)
    label = tf.cast(Features['label'], tf.float32)

    # 第六步：生成Batch数据 generate batch
    if shuffle_batch:  
        #打乱数据顺序，随机取样
        sample_single, label_single = tf.train.shuffle_batch([sample, label],
                                                 batch_size=2,
                                                 capacity=200000,
                                                 min_after_dequeue=10000,
                                                 num_threads=1,
                                                 enqueue_many=if_enq_many)
    else:  
        #如果不打乱顺序则用tf.train.batch(), 输出队列按顺序组成Batch输出
        sample_single, label_single = tf.train.batch([sample, label],
                                                batch_size=2,
                                                capacity=200000,
                                                min_after_dequeue=10000,
                                                num_threads=1,
                                                enqueue_many = if_enq_many)
    return sample_single, label_single
```

**(2) tf.parse_example()读取数据**

```python
# 第一步： 建立文件名队列
filename_queue = tf.train.string_input_producer([tfrecord_path],num_epochs=3)
def read_parse(filename_queue, shuffle_batch, if_enq_many):
    # 第二步： 建立阅读器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 第三步： 设置shuffle_batch
    if shuffle_batch:
        batch = tf.train.shuffle_batch([serialized_example],
                               batch_size=3,
                               capacity=10000,
                               min_after_dequeue=1000,
                               num_threads=1,
                               enqueue_many=if_enq_many) #为了评估enqueue_many的作用

    else:
        batch = tf.train.batch([serialized_example],
                               batch_size=3,
                               capacity=10000,
                               num_threads=1,
                               enqueue_many=if_enq_many)
        
    # 第四步：根据写入时的格式建立相对应的读取features
    features = {
        'sample': tf.FixedLenFeature([5], tf.int64),  # 如果不是标量，要说明数组的长度
        'label': tf.FixedLenFeature([], tf.int64)
    }
    # 第五步： 用tf.parse_example()解析多个EXAMPLE PROTO
    Features = tf.parse_example(batch, features)

    # 第六步：对数据进行后处理
    samples_parse= tf.cast(Features['sample'], tf.float32)
    labels_parse = tf.cast(Features['label'], tf.float32)
    return samples_parse, labels_parse
```



`tf.FixedLenFeature` 返回一个定长的tensor，通常用来解析单个元素的变量或固定长度list的变量，同时需要指明变量的类型；

`tf.VarLenFeature` 返回一个不定长的sparse tensor，通常用来处理大小不同的图片或长度不同的问题数据，也是需要指明变量的类型。






























