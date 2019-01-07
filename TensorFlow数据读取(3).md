# 数据集(Dataset)

除了使用队列外，TensorFlow还提供了一套更加高层的数据处理框架。在新的框架中，每一个数据来源被抽象成一个"数据集"，以此数据集为基础可以方便地进行batch、shuffle等操作。

在数据集框架中，每一个数据集代表一个数据来源：可能来自一个张量，或者一个文件（如文本文件、TFRecord文件或者其他类型）。

### 从内存中创建数据集

`tf.data.Dataset.from_tensor_slices`函数可以既可以处理tensor参数，也可以处理np.array参数。下面是使用np.array的方式。

```PYTHON
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0]))
dataset = dataset.batch(2)
itr = dataset.make_one_shot_iterator()
element = itr.get_next()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    try:
        while True:
            print(sess.run(element))
    except tf.errors.OutOfRangeError:
        print("All Data Done!")
    finally:
        pass
```

`tf.data.Dataset.from_tensor_slices`还有其他功能，切分传入Tensor的第一个维度，生成相应的dataset。代码如下所示。

```PYTHON
dataset = tf.data.Dataset.from_tensor_slices({'f': tf.random_uniform([3, 4]), 'l': np.array([1, 0, 1])})
#方式二
#dataset = tf.data.Dataset.from_tensor_slices((tf.random_uniform([3, 4]), np.array([1, 0, 1])))

itr = dataset.make_initializable_iterator()#这里不再使用make_one_shot_iterator
example = itr.get_next()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    sess.run(itr.initializer) #需要初始化

    try:
        while True:
            print(sess.run(example))
    except tf.errors.OutOfRangeError:
        print("All done")
    finally:
        pass
```

这段代码需要注意的是没有使用`make_one_shot_iterator` 而是使用了`make_initializable_iterator` 是因为我们在`from_tensor_slices`中使用了`tf.random_uniform`变量需要对其进行初始化，不然会提示错误。`make_one_shot_iterator`不需要进行初始化，但是`make_initializable_iterator`必须初始化。

### 从文件中创建数据集

+ `tf.data.TextLineDataset()`函数的输入是一个文件的列表，输出是一个dataset。dataset中的每一个元素对应了文件中的一行。可以使用这个函数读取文本文件。
+ `tf.data.FixedLengthRecordDataset()`函数的输入是一个文件列表和一个record_bytes，之后dataset中每个元素都是文件中固定字节数record_bytes的内容。通常用来读取二进制形式保存的文件。
+ `tf.data.TFRecordDataset()`用来读取TFRecord文件，dataset的每个元素都是一个TFExample。



### 元素变换

Dataset支持一类特殊的操作：**Transformation** 。一个Dataset通过Transformation可以变成另一个新的Dataset。常用的变换操作有：

####  (1)map

map参数接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset。

#### (2)shuffle

shuffle的功能是将dataset中的元素打乱，参数buffer_size等效于`tf.train.shuffle_batch`的min_after_dequeue参数。shuffle操作在内部使用一个缓冲区保存buffer_size条数据，每读入一条数据时从这个缓冲区中随机选择一条数据进行输出。缓冲区的大小越大，随机性能越好，但占用的内存也越多。

#### (3)batch

batch将多个元素组成成一个batch，输送给模型进行训练。最后一个batch的大小可能是小于batch_size。如果要保证每个batch的大小都是一样的，这时候需要舍弃到最后一个batch，可以使用`tf.contrib.data.batch_and_drop_remainder(self.batch_size)`

#### (4)repeat

repeat将整个数据复制多分，其中每一份数据被称为一个epoch。需要注意的是如果数据集在repeat之前进行了shuffle操作，输出的每个epoch中随机shuffle的结果并不会相同。如果直接调用repeat()没有指定num_repeat那么生成的序列会无限重复下去，没有结束，是不会抛出`tf.errors.OutOfRangeError`异常的。

#### (5)prefetch

prefetch提供了一个 software pipelining 机制，这个机制解耦了数据产生和数据消耗。尤其是，这个机制使用一个后台线程和一个内部缓存区，在数据被请求前，去从数据数据集中预加载一些数据。

```PYTHON
dataset.shuffle(self.dataset_num)
       .map(parse_fn, num_parallel_calls=8)           
       .batch(self.batch_size)
       .prefetch(self.batch_size)  
       .repeat(self.repeat_count)
        
#################################################
dataset.shuffle(self.dataset_num)
       .map(parse_fn, num_parallel_calls=8)
       .apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
       .prefetch(self.batch_size)
       .repeat(self.repeat_count)
```









































