# To classify images using a recurrent network, we consider every image
# row as a sequence of pixels. Because MNIST image shape is 28*28px,
# we will handle then 28 sequences of 28 steps for every sample.

import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist',one_hot=True)

# Training parameters
lr = 0.01
train_steps = 1000
batch_size = 128
display_step = 200

# Network parameters
num_input = 28 # MNIST data input (img shape: 28*28px), 特徵數量
timesteps = 28 # timestep, 每張圖片28個rows
num_hidden = 128 # hidden layer num of features, 128個neurons
num_classes = 10 # MNIST total classes(0~9 digits), 標籤類別數量

# tf Graph input
X = tf.placeholder(tf.float32,shape=[None, timesteps, num_input])
Y = tf.placeholder(tf.float32,shape=[None, num_classes])

# Define weights
weights = {
    'out' : tf.Variable(tf.random_normal([num_hidden,num_classes]))
}
biases = {
    'out' : tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Current data input shape: (batch_size, timesteps, num_input)
    # Required shape: 'timesteps' tensor list of shape (batch_size,num_input)

    # Unstack to get a list of 'timesteps' tensors list of shape (batch_size,num_input)
    x = tf.unstack(value=x, num=timesteps, axis=1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden,forget_bias=1)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    # x.shape=[timesteps,batch_size,num_input]
    # 使用rnn.static_rnn的話, 輸入是一個list, 而len(list)=timesteps, list[0].shape=[batch_size,num_input]
    # 而輸出也是一個list, len(list)=timesteps, list[0].shape=[batch_size,num_hidden]

    return tf.matmul(outputs[-1],weights['out']) + biases['out']
    # 而outputs[-1]則是取最後一個timestep的output

logits = RNN(X,weights,biases)
prediction = tf.nn.softmax(logits=logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
# tf.equal判斷兩者每個元素是否相等, 返回一個bool的tensor
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
# tf.cast將數據格式轉換為指定的dtype, 如上式從tf.bool轉成tf.float32
# 接著計算總平均, 如果全部皆為1, accuracy就是100%

with tf.Session() as sess:
    init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init)

    for step in range(1, train_steps+1): # 更新1000次
        batch_x, batch_y = mnist.train.next_batch(batch_size) # 每次拿128個出來訓練
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size,timesteps,num_input))
        # 因為mnist中每張圖片是784的vector, 需轉回成28*28px才能依照每次time_step丟進去訓練
        # 而共有55000張, 每次訓練取batch_size張出來訓練

        # Run optimization op (backprop)
        sess.run(train_op,feed_dict={X: batch_x,Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op,accuracy],feed_dict={X: batch_x,Y: batch_y})
            print('Step ' + str(step) + ', Minibatch Loss= ' + \
                  '{:.4f}'.format(loss) + ", Training Accuracy= " + \
                  '{:.3f}'.format(acc))

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1,timesteps,num_input))
    # 預測時不需要轉為[timesteps,batch_size,num_input], 只需要依照placeholder假設的shape即可
    test_label = mnist.test.labels[:test_len]
    print('Testing Accuracy: ', \
          sess.run(accuracy,feed_dict={X: test_data,Y: test_label}))

# https://www.zhihu.com/question/52200883/answer/136317118
# 設定Minibatch後, 傳入model的是一個batch的數據
# (一個batch數據forward得到predictions,計算loss,然後backpropagation更新參數)
# 且每一個batch的sequence一定是相同長度的
# 探討tf.nn.dynamic_rnn與rnn.static_rnn的差別
# 而dynamic_rnn實現的是讓不同迭代傳入的batch可以是不同長度的sequence
# 但同一次迭代一個batch內部的長度仍然是固定的(sequence長度:num_input)
# 例如說第一個timesteps的shape=[batch_size,10]
# 第二個timesteps的shape=[batch_size,12]...

# 但是rnn不能這樣，而是要求每個timesteps的shape都必須為[batch_size,max_num_input]