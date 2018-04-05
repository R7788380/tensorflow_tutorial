# 快速註釋, command + /
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# seed
tf.set_random_seed(1)
np.random.seed(1)

# Parameter
batch_size = 100
time_step = 28
input_size = 28
lr = 0.01

# Data
mnist = input_data.read_data_sets('./mnist',one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# Plot one example
plt.imshow(mnist.train.images[0].reshape((28,28)),cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()
plt.close()
print(mnist.train.images[0].shape) # (,784), 每張圖片被拉成784維的vector
print(mnist.train.labels[0].shape) # (,10), one-hot encoding, 共0~9, 10個數字

# Placeholder
tf_x = tf.placeholder(tf.float32,[None,time_step * input_size])
tf_y = tf.placeholder(tf.int32,[None,10])
image = tf.reshape(tf_x,[-1,time_step,input_size])
# 為了讓dynamic_rnn使用, shape轉換成[batch_size, time_step, input_size]

# Dynamic_rnn
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units = 64)
# 使用tf.nn.dynamic_rnn, 輸入為[batch_size,time_step,input_size]
# 輸出為[batch_size,time_step,n_hidden]
outputs, (h_c,h_n) = tf.nn.dynamic_rnn(rnn_cell,
                                       image,
                                       initial_state=None,
                                       dtype=tf.float32,
                                       time_major=False) # (batch_size,time_step,input_size)
output = tf.layers.dense(inputs=outputs[:,-1,:],units=10)
# outputs[:,-1,:]表示取最後一步的output作為output layer的input

# Loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)

# Train operation
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# Accuracy
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1))[1]


def plot_images_label_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12,24)
    if num > 25:
        num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx,:].reshape((28,28)), cmap='gray')
        title = 'actual: %i,prediction= %i' % (labels[idx],prediction[idx])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()

# Train
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
# 因為mnist.train.next_batch在next時需要記住上一次的位置, 所以使用global_variables
# 而上面mnist.test.images挑出前2000筆是local_variables, 所以global與local都需要initialize
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(1000):
        b_x, b_y = mnist.train.next_batch(batch_size)
        _, loss_ = sess.run([train_op,loss],feed_dict={tf_x: b_x, tf_y: b_y})
        if step % 50 == 0:
            accuracy_ = sess.run(accuracy,feed_dict={tf_x: test_x,tf_y: test_y})
            print('train loss: %.4f' % loss_,'test accuracy: %.2f' % accuracy_)
        # print prediction and images
    # 注意縮排, 如果往後tab會造成每50個結果就print一次
    test_output = sess.run(output,feed_dict= {tf_x: test_x}) # 輸出是一個0~9共10個位置的概率
    # 這裏產生test的預測結果
    pred_y = np.argmax(test_output,axis=1) # 挑出預測中概率最大的位置
    print(pred_y[:10],'prediction number') # 打印出前10筆
    real_num = np.argmax(test_y,axis=1) # 挑出真實值概率為1的位置, 因為read_data時已經轉換成one-hot encoding
    print(real_num[:10],'real number')
    # 顯示前10張圖片與預測結果
    plot_images_label_prediction(test_x,labels=real_num,prediction=pred_y,
                                 idx=0,num=10)




