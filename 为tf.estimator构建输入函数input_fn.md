## 为`tf.estimator`构建输入函数input_fn

`input_fn`用于数据预处理并将数据输入到模型中，可以在神经网络中提供训练、验证和预测数据。以下代码说明了输入函数的基本框架：

```PYTHON
def my_input_fn():
    #preprocess your data here ...
    
    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels
```

这个函数的主体包括了对数据的预处理逻辑，如数据清洗和特征变换等。函数必须返回两个值，包括了最终的特征数据和标签数据：

+ `feature_cols`：一个包含键值对的字典，将特征列名称映射到一个Tensor或SparseTensor中。（这里的Tensor或SparseTensor包含了与之对应关联的特征数据）；
+ `labels`：一个包含了标签数据的Tensor，即模型需要预测的值。



### 特征数据转换为张量

特征数据/标签数据如果是一个Python数组或存储在Pandas DataFrame或Numpy的数组中，可以直接使用TensorFlow的内置方法进行读取并构造input_fn：

```PYTHON
import numpy as np
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x':np.array(x_data)},
    y=np.array(y_data),
    batch_size=128,
    num_epochs=1,
    shuffle=None,
    queue_capacity=1000,
    num_threads=1,
    target_column='target'
)


import pandas as pd
my_input_fn=tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({'x':x_data}),
    y=pd.Series(y_data),
    batch_size=128,
    num_epochs=1,
    shuffle=None,
    queue_capacity=1000,
    num_threads=1,
    target_column='target'
)
```



### input_fn数据传递给模型

要将数据提供给模型进行训练，只需将创建的输入函数`my_input_fn`传递给模型`train`作为`input_fn`的参数值即可，如：

```PYTHON
model.train(input_fn=my_input_fn,steps=1000)
```

需要注意的是`input_fn`参数必须接受函数对象（即input_fn=my_input_fn）而不是函数调用的返回值，注意不是input_fn=my_input_fn()。这就意味着在`train`调用中，如果尝试传递参数给`my_input_fn`，如下面的代码，会导致一个TypeError：

```PYTHON
model.train(input_fn=my_input_fn(training_set),steps=1000)
```



对于带参数的输入函数，其处理方法主要以下四种方案：

（1）使用一个不带参数的包装函数，并用它来调用所需的参数的输入函数，即：

```PYTHON
def my_input_fn(data_set):
    ...
    
def my_input_fn_training_set():
    return my_input_fn(data_set)

model.train(input_fn=my_input_fn_training_set,steps=1000)
```

（2）使用一个带参数的包装函数，里面封装`input_fn`最后结果注意返回`input_fn`

```PYTHON
def train_input_fn(data_set):
    def input_fn():
        ...
        
        return features,label
    
    return input_fn

model.train(input_fn=train_input_fn(data_set),steps=1000)
```

（3）使用Python的`functools.partial`函数来构建一个固定了所有参数值的新函数对象

```PYTHON
def my_input_fn(data_set):
    ...

model.train(input_fn=functools.partial(my_input_fn,data_set=training_set),steps=1000)
```

（4）使用lambda进行包装

```python
def my_input_fn(data_set):
    ...
    
model.train(input_fn=lambda:my_input_fn(data_set),steps=1000)
```



采用`input_fn`设计输入管道的优势在于增强了代码的可维护性，针对不同数据数据集如train_set、test_set和predict_set可以统一设计一个`input_fn` 。

```PYTHON
model.train(input_fn=lambda:my_input_fn(data_set),steps=1000)
model.evaluate(input_fn=lambda:my_input_fn(test_set))
model.predict(input_fn=lambda:my_input_fn(predict_set))
```



一个完整的示例，读取boston房价的回归案例：

```python
COLUMNS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'medv']
FEATURES = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio']
LABEL = 'medv'


def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({col: data_set[col].values for col in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle
    )


def main(_):
    training_set = pd.read_csv('boston_train.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv('boston_test.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
    predict_set = pd.read_csv('boston_predict.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)

    feature_col = [tf.feature_column.numeric_column(k) for k in FEATURES]

    model = tf.estimator.DNNRegressor(feature_columns=feature_col,
                                      hidden_units=[10, 256, 10], model_dir='model/')

    model.train(input_fn=get_input_fn(training_set), steps=5000)

    evaluate = model.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False),
                              steps=1)
    print(evaluate)

    predictions = model.predict(input_fn=get_input_fn(predict_set, num_epochs=1, shuffle=False))
    for pred in predictions:
        print(pred)


if __name__ == '__main__':
    tf.app.run()
```































