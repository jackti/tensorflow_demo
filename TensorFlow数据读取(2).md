# 批处理

经过了前面的数据读取流程之后，现在需要有另一个队列来批量执行输入样本训练，评估或推断。常用的有两个函数：

> `tf.train.batch()`
>
> `tf.train.shuffle_batch()`

使用说明

```python
tf.train.batch(
    tensors,
    batch_size,
    num_threads=1,
    capacity=32,
    enqueue_many=False,
    shapes=None,
    dynamic_pad=False,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
)
```

`tf.train.batch`函数将会使用一个队列，函数读取一定数量的tensors送入队列，然后每次从中选取batch_size个tensors组成一个新的tensors返回。batch_size指定了批处理的大小。capacity参数决定了队列的长度。

num_threads决定了有多少个线程运行入队操作，如果设置的超过一个线程，那么它们会从不同文件的不同位置同时读取，可以更加充分的混合训练样本。

如果 enqueue_many=False，则 tensor 表示单个样本.对于 shape 为 [x, y, z] 的输入 tensor，该函数输出为，shape 为 [batch_size, x, y, z] 的 tensor。如果 enqueue_many=True，则 tensors 表示 batch 个样本，其中，第一维表示样本的索引，所有的 tensors 都在第一维具有相同的尺寸.对于 shape 为 [*, x, y, z] 的输入 tensor，该函数输出为，shape 为 [batch_size, x, y, z] 的 tensor。

enqueue_many主要是设置tensor中的数据是否能重复,如果想要实现同一个样本多次出现可以将其设置为:”True”,如果只想要其出现一次,也就是保持数据的唯一性,这时候我们将其设置为默认值:”False”

当allow_smaller_final_batch为True时，如果队列中的张量数量不足batch_size，将会返回小于batch_size长度的张量，如果为False，剩下的张量会被丢弃。



```PYTHON
tf.train.shuffle_batch(
    tensors,
    batch_size,
    capacity,
    min_after_dequeue,
    num_threads=1,
    seed=None,
    enqueue_many=False,
    shapes=None,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
)
```

`tf.train.shuffle_batch`函数类似于上面的`tf.train.batch()`，同样创建一个队列，主要区别是会首先把队列中的张量进行乱序处理，然后再选取其中的batch_size个张量组成一个新的张量返回。

capacity参数依然为队列的长度，建议capacity的取值如下：

> ​                   min_after_dequeue + (num_threads + a small safety margin) * batch_size

min_after_dequeue这个参数的含义在队列中，做dequeue（取数据）的操作后，线程要保证队列中至少剩下min_after_dequeue个数据。如果min_after_dequeue设置的过少，则即使shuffle为True，也达不到好的混合效果。但是min_after_dequeue也不能设置的太大，这样会导致队列填充的时间变长，尤其是在最初的装载阶段，会花费比较长的时间。

一个批处理的例子：

```python
def read_my_file_format(filename_queue):
  reader = tf.SomeReader()
  key, record_string = reader.read(filename_queue)
  example, label = tf.some_decoder(record_string)
  processed_example = some_processing(example)
  return processed_example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue 越大意味着随机效果越好但是也会占用更多的时间和内存
  # capacity 必须比 min_after_dequeue 大
  # 建议capacity的取值如下：
  # min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

  return example_batch, label_batch
```





## 多样本和多阅读器

#### 单个Reader，单个样本

```PYTHON
filename_queue = tf.train.string_input_producer(['A.csv', 'B.csv', 'C.csv'], num_epochs=1, shuffle=False)
reader = tf.TextLineReader()
_, value = reader.read(filename_queue)
feature, label = tf.decode_csv(value, [['null'], ['null']])

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        for i in range(10):
            print(sess.run([feature, label]))
    except tf.errors.OutOfRangeError:
        print("All done!")
    coord.request_stop()
    coord.join(threads)
    
##########################
[b'Alpha1', b'A1']
[b'Alpha2', b'A2']
[b'Alpha3', b'A3']
[b'Bee1', b'B1']
[b'Bee2', b'B2']
[b'Bee3', b'B3']
[b'Sea1', b'C1']
[b'Sea2', b'C2']
[b'Sea3', b'C3']
All done!
```



#### 单个Reader，多个样本

```PYTHON
filename_queue = tf.train.string_input_producer(['A.csv', 'B.csv', 'C.csv'], num_epochs=10, shuffle=False)
reader = tf.TextLineReader()
_, value = reader.read(filename_queue)
feature, label = tf.decode_csv(value, [['null'], ['null']])
batch_feature, batch_label = tf.train.batch([feature, label], batch_size=5, num_threads=2)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        for i in range(10):
            print(sess.run([batch_feature, batch_label]))
    except tf.errors.OutOfRangeError:
        print("All done!")
    coord.request_stop()
    coord.join(threads)
```



#### 多个reader，多个样本

```PYTHON
filename_queue = tf.train.string_input_producer(['A.csv', 'B.csv', 'C.csv'], num_epochs=10, shuffle=False)
reader = tf.TextLineReader()
_, value = reader.read(filename_queue)
#Reader设置为2
example_list = [tf.decode_csv(value, [['null'], ['null']]) for _ in range(2)]
# 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程
batch_feature, batch_label = tf.train.batch_join(example_list, batch_size=5)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
        for i in range(10):
            print(sess.run([batch_feature, batch_label]))
    except tf.errors.OutOfRangeError:
        print("All done!")
    coord.request_stop()
    coord.join(threads)
```

`tf.train.batch`和`tf.train.shuffle_batch`函数是单reader读取，但是可以多线程，通过参数num_threads来设置多线程。`tf.train.batch_join`和`tf.train.shuffle_batch_join`可设置多Reader读取，每个Reader使用一个线程。关于两者的效率，单Reader时2个线程就达到了速度极限。多Reader时2个Reader就达到了极限。所以并不是线程越多越快，甚至更多的线程反而会使效率下降。





































