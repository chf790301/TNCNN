import time
import datetime
import os
import numpy as np
# import tensorflow as tf # tensorflow 1.x

# tensorflow 2.x
import tensorflow as tf


import matplotlib.pyplot as plt

import readfile_3

def focal_loss(pred, y, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    zeros = tf.zeros_like(pred, dtype=pred.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = tf.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = tf.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def batch_norm(x,epsilon=1e-5, momentum=0.9,train=True, name="batch_norm"):
    with tf.variable_scope(name):
        epsilon = epsilon
        momentum = momentum
        name = name
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,scale=True, is_training=train,scope=name)

train_ep_record = open('E:\陈老师\河南\数据列表\河南\\200km\河南及全国200km\stalta找起点\\ep_train.txt')
train_ep_record_read = train_ep_record.readlines()
train_ep_record.close()

train_ss_record = open('E:\陈老师\河南\数据列表\河南\\200km\河南及全国200km\stalta找起点\\ss_train.txt')
train_ss_record_read = train_ss_record.readlines()
train_ss_record.close()

test_ep_record = open('E:\陈老师\河南\数据列表\河南\\200km\河南及全国200km\stalta找起点\\ep_test.txt')
test_ep_record_read = test_ep_record.readlines()
test_ep_record.close()

test_ss_record = open('E:\陈老师\河南\数据列表\河南\\200km\河南及全国200km\stalta找起点\\ss_test.txt')
test_ss_record_read = test_ss_record.readlines()
test_ss_record.close()

TRAINNUM = 1000
batch_size = 64 # 5+-各64个单分量
test_size = 4 # 5+-各16个单分量


# 保存模型及运行过程信息
model_info = '64-ts35-70s-河南三分量+全国200km经人工筛选sta-1000'
training_time = datetime.datetime.now().strftime("%y-%m-%d+%h-%M-%S")
if not os.path.exists(r'./svntin'):
    os.makedirs(r'./svntin')
strsvin = r'./svntin' + '/' + model_info + training_time   #生成模型位置
if not os.path.exists(strsvin):
    os.makedirs(strsvin)
fsvin = open(strsvin + '/training_info.txt', 'w')
frecord = open(strsvin + '/training_record.txt', 'w')
start_time = time.time() # 返回浮点秒数
fsvin.write('1000次迭代，每batch各64个五级上下样本')
fsvin.write('run in ' + str(start_time) + ' EEW.py\n')
fsvin.write('train num is ' + str(TRAINNUM) + '\n')
fsvin.write('batch size is ' + str(batch_size) + '\n')
fsvin.write('test size is ' + str(test_size) + '\n')


sess = tf.InteractiveSession()  # 创建session

def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
def avg_pool_6x6(x):
   return tf.nn.avg_pool(x,strides=[1, 1, 600, 1],ksize=[1, 1, 600, 1],padding="SAME")

x = tf.placeholder(tf.float32, [None, 1, 7000, 3], name="x-sword")
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32, name='kp')

xs = tf.reshape(x, [-1, 1, 7000, 3])


W_conv1 = weight_variable([1, 5, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)
m_pool1 = max_pool_2x2(h_conv1) # 3500

W_conv2 = weight_variable([1, 1, 16, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(m_pool1, W_conv2) + b_conv2)
m_pool2 = max_pool_2x2(h_conv2) # 1750

W_conv3 = weight_variable([1, 3, 16, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(m_pool2, W_conv3) + b_conv3)
m_pool3 = max_pool_2x2(h_conv3) # 875

W_conv4 = weight_variable([1, 1, 32, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(m_pool3, W_conv4) + b_conv4)
m_pool4 = max_pool_2x2(h_conv4) # 438

W_conv5 = weight_variable([1, 3, 32, 64])
b_conv5 = bias_variable([64])
h_conv5 = tf.nn.relu(conv2d(m_pool4, W_conv5) + b_conv5)
m_pool5 = max_pool_2x2(h_conv5) # 219

W_conv6 = weight_variable([1, 1, 64, 64])
b_conv6 = bias_variable([64])
h_conv6 = tf.nn.relu(conv2d(m_pool5, W_conv6) + b_conv6)
m_pool6 = max_pool_2x2(h_conv6) # 110


W_fc1 = weight_variable([110 * 64, 2]) # 减小全连接层的参数个数[512,128/64/32]
b_fc1 = bias_variable([2])
h_conv10_flat = tf.reshape(m_pool6, [-1, 110 * 64])
h_conv10_drop = tf.nn.dropout(h_conv10_flat, keep_prob)
y_conv = tf.nn.softmax(tf.matmul(h_conv10_drop, W_fc1) + b_fc1)


print(tf.trainable_variables())

tf.add_to_collection("sword", y_conv)

# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nt_hpool6_flat,labels=tf.cast(ys,tf.int64))
# cross_entropy = -tf.reduce_sum(ys * tf.log(tf.clip_by_value(y_conv, 1e-8, 1.0))) # 带截断的交叉熵损失
# cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))

cross_entropy = focal_loss(pred=y_conv, y=ys)

# train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cross_entropy)
# train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001, momentum=0.9, decay=0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()

test_acc_max = 0.0
cost_x = []
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for i in range(TRAINNUM + 1):

    X_train, Y_train = readfile_3.load_train_data(train_ep_record_read, train_ss_record_read, batch_size)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # print(i, X_train.shape, Y_train.shape)

    if i % 5 == 0:
        cost_x.append(i)
        train_accuracy = accuracy.eval(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.5})
        train_entropy = cross_entropy.eval(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.5})
        train_loss.append(train_entropy)
        train_acc.append(train_accuracy)
    if i % 5 == 0:
        X_test, Y_test = readfile_3.load_test_data(test_ep_record_read, test_ss_record_read, test_size)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        # print(X_test.shape, Y_test.shape)
        test_entropy = cross_entropy.eval(feed_dict={x: X_test, ys: Y_test, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x: X_test, ys: Y_test, keep_prob: 1.0})
        test_loss.append(test_entropy)
        test_acc.append(test_accuracy)
        print("step %d, training loss %g, training accuracy %g" % (i, train_entropy, train_accuracy))
        print("step %d, testing loss %g, testing accuracy %g" % (i, test_entropy, test_accuracy))
        # print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, ys: Y_test, keep_prob: 1.0}))
        frecord.write("step %d, training loss %g, training accuracy %g" % (i, train_entropy, train_accuracy) + '\n')
        frecord.write("step %d, testing loss %g, testing accuracy %g" % (i, test_entropy, test_accuracy) + '\n')

        if test_accuracy > 0.9 and train_accuracy > 0.9:
            test_acc_madel = strsvin + '\\'+ str(test_accuracy)
            if not os.path.exists(test_acc_madel):
                os.makedirs(test_acc_madel)
            saver = tf.train.Saver()
            saver.save(sess, strsvin + '\\' + str(test_accuracy) + '/model.ckpt')

    train_step.run(feed_dict={x: X_train, ys: Y_train, keep_prob: 0.5})


plt.xlabel('Iterations')
plt.ylabel('Training loss')
plt.plot(cost_x, train_loss)
plt.savefig(strsvin + '\\' + 'Training_loss.jpg')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Training accuracy')
plt.plot(cost_x, train_acc)
plt.savefig(strsvin + '\\' + 'Training_accuracy.jpg')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Testing loss')
plt.plot(cost_x, test_loss)
plt.savefig(strsvin + '\\' + 'Testing_loss.jpg')
plt.show()

plt.xlabel('Iterations')
plt.ylabel('Testing accuracy')
plt.plot(cost_x, test_acc)
plt.savefig(strsvin + '\\' + 'Testing_accuracy.jpg')
plt.show()

if not os.path.exists(strsvin + '/final'):
    os.makedirs(strsvin + '/final')
saver = tf.train.Saver()
saver.save(sess, strsvin + '/final' + '/model.ckpt')

end_time = time.time()
fsvin.write('endtime is ' + str(end_time) + '\n')
fsvin.write('costtime is ' + str(end_time - start_time) + '\n')
fsvin.close()
frecord.close()

print("训练完毕")