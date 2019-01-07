TensorFlow设计模块

```PYTHON
def get_weight(shape,regularizer):
    w = tf.Variable(...)
    if regularizer is not None:
      tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(...)
    return b

def forward(x, regularizer):
    w = 
    b = 
    ...
    y = 
    return y


def backward():
    x = tf.placehold(...)
    y_ = tf.placehold(...)
    y = forward(x,REGULARIZER)
    
    
    ###################正则化损失函数#######################
    #回归问题
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    #分类问题
    loss_ce = tf.spare_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    #损失函数加入正则化
    loss = loss_mse/loss_ce+tf.add_n(tf.get_collection('losses'))
    
    ###################指数衰减学习率#######################
    global_step = tf.Varible(0,trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                  			   总样本数/BATCH_SIZE,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)
    train_step = tf.train.SGD(learning_rate).minimize(loss,global_step)
    
    ###################滑动平均#######################
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')
    
    
    ###################准确率计算#######################
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initalizer()
        sess.run(init)
        
        for i in range(STEPS):
            ######训练过程中的中间结果######
            _,loss_val,acc,global_step 	=
            sess.run([train_step,loss,accuracy,global_step],feed_dict={x: ,y: })
            
            if i%100 == 0:
                saver.save("path",global_step= global_step)
                print()
                

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placehold(...)
        y_ = tf.placehold(...)
        y = forward(x,REGULARIZER)
        
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state('path')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    acc = sess.run(accuracy,feed_dict={x: ,y_: })
                    print()
                   
               
  
```



