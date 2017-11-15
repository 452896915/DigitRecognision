# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import scipy.io

from tensorflow.python import debug as tf_debug

from tensorflow.examples.tutorials.mnist import input_data

# 训练图像数据
mat = scipy.io.loadmat('database/format2/train_32x32.mat')
print mat.keys()

vals_train = mat.get("X")
labels_train = mat.get("y")

# 测试图像数据
mat_test = scipy.io.loadmat('database/format2/test_32x32.mat')
print mat_test.keys()

vals_test = mat_test.get("X")
labels_test = mat_test.get("y")

train_photo_cnt = 10000
train_lists = [] # 训练图片
train_labels = [] # 训练图片的标记

test_photo_cnt = 1000
test_lists = [] # 测试图片
test_labels = [] # 测试图片的标记


def gerPreprocessData(vals, labels, photo_cnt):
    photo_lists = []
    label_lists = []
    for i in range(photo_cnt):  # 遍历73257张图片中的每一张 np.shape(vals)[3]
        red_vals = vals[:, :, 0, i]
        green_vals = vals[:, :, 1, i]
        blue_vals = vals[:, :, 2, i]
        label = labels[i]

        print label

        w = np.shape(vals)[0]
        h = np.shape(vals)[1]
        # gray_vals = [[0 for x in range(w)] for y in range(h)]

        gray_vals_floats = red_vals * 0.3 + green_vals * 0.59 + blue_vals * 0.11 + 0.50  # uint8最高256，不要超过，否则出错
        gray_vals = np.array(gray_vals_floats, dtype="uint8")
        # for x in range(w):
        #     for y in range(h):
        #         # 计算一个像素的灰度值
        #         gray_vals[x][y] = (red_vals[x][y] * 30 + green_vals[x][y] * 59 + blue_vals[x][y] * 11 + 50) / 100

        photo_lists.append(gray_vals)
        label_lists.append(tf.one_hot([label[0]], depth=10))
    return photo_lists, label_lists

(train_lists, train_labels) = gerPreprocessData(vals_train, labels_train, train_photo_cnt)
(test_lists, test_labels) = gerPreprocessData(vals_test, labels_test, test_photo_cnt)

n_classes = 10
length = 32
width = length
height = length
x = tf.placeholder("float", shape=(width, height))
y = tf.placeholder("float", shape=(1, n_classes))

is_training = tf.placeholder(tf.bool)


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def CNN(inputs, is_training=True):
    x = tf.reshape(inputs, [-1, width, height, 1])  # NHWC  N=1 Sample的数量 C=1 一个通道，灰度值
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    init_func = tf.truncated_normal_initializer(stddev=0.01) # 正太分布初始化

    # 1） input： 32 * 32 * 1 out: 32 + 5 -1 = 36  36 * 36 * 16 = 36 * 36 * 2^4
    net = slim.conv2d(x, 16, [5, 5], padding='SAME'
                      , activation_fn=lrelu
                      , weights_initializer=init_func
                      , normalizer_fn=slim.batch_norm
                      , normalizer_params=batch_norm_params
                      , scope='conv0')

    # in: 36 * 36 * 16  out: 18 * 18 * 16
    net = slim.max_pool2d(net, [2, 2], scope='pool0')

    # 2） in: 18 * 18 * 16 out: 18 + 5 - 1 = 22  22 * 22 * 16 * 32 = 22 * 22 * 2^9
    net = slim.conv2d(net, 32, [5, 5], padding='SAME'
                      , activation_fn=lrelu
                      , weights_initializer=init_func
                      , normalizer_fn=slim.batch_norm
                      , normalizer_params=batch_norm_params
                      , scope='conv1')
    # in: 22 * 22 * 2^9 out: 11 * 11 * 2^9
    net = slim.max_pool2d(net, [2, 2], scope='pool1')

    # 3） in: 11 * 11 * 2^9 out: 11 + 5 -1 = 15 15 * 15 * 2^9 * 2^6 = 15 * 15 * 2^15
    net = slim.conv2d(net, 64, [5, 5], padding='SAME'
                      , activation_fn=lrelu
                      , weights_initializer=init_func
                      , normalizer_fn=slim.batch_norm
                      , normalizer_params=batch_norm_params
                      , scope='conv2')
    # in: 15 * 15 * 2^15  out: 8 * 8 * 2^15
    net = slim.max_pool2d(net, [2, 2], scope='pool2')

    # 把矩阵flattern成一维的，[batch_size, k]
    net = slim.flatten(net, scope='flatten3')

    # 第一个全连接层
    net = slim.fully_connected(net, 1024
                               , activation_fn=lrelu
                               , weights_initializer=init_func
                               , normalizer_fn=slim.batch_norm
                               , normalizer_params=batch_norm_params
                               , scope='fc4')
    net = slim.dropout(net, keep_prob=0.7, is_training=is_training, scope='dr')

    # 第二个全连接层
    out = slim.fully_connected(net, n_classes
                               , activation_fn=None, normalizer_fn=None, scope='fco')
    return out


print ("神经网络准备完毕")

# PREDICTION
pred = CNN(x, is_training)
#ppred = tf.nn.softmax(pred) # [N, 10]  softmax归一化，不添加这一句导致cost计算为nan

# LOSS AND OPTIMIZER
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) # 计算输出和标记结果的交叉熵作为损失函数
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

corr = tf.equal(tf.arg_max(pred, 1), tf.argmax(y, 1))  # 按行取最大值所在的位置，比较两个位置是否相同
accr = tf.reduce_mean(tf.cast(corr, "float"))  # 准确度

# INITIALIZER
init = tf.global_variables_initializer()
sess = tf.Session()

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

sess.run(init)
print ("FUNCTIONS READY")

# 检查变量
print ("=================== TRAINABLE VARIABLES ===================")
t_weights = tf.trainable_variables()
var_names_list = [v.name for v in tf.trainable_variables()]
for i in range(len(t_weights)):
    wval = sess.run(t_weights[i])
    print ("[%d/%d] [%s] / SAHPE IS %s" % (i, len(t_weights), var_names_list[i], wval.shape,))

# 存储模型路径
savedir = "model_out/"
saver = tf.train.Saver(max_to_keep=100)
save_step = 4
if not os.path.exists(savedir):
    os.makedirs(savedir)
print ("SAVER READY")

# PARAMETERS
training_epochs = 50  # 在整个训练集上过多少遍
batch_size = 1 # 50 每次处理训练集的一个batch的数量
display_step = 1

val_acc = 0
val_acc_max = 0
current_best_accuracy = 0.0

# OPTIMIZE
currentTime = time.time()
total_cost = 0.
total_cnt = 0
for epoch in range(training_epochs): # 循环处理所有训练集多次
    total_batch = train_photo_cnt    #int(mnist.train.num_examples / batch_size)  # 训练数据集分割成若干个输入batch，一次处理一个batch
    # 循环处理所有训练集一次 start
    for i in range(total_batch):
        batch_xs = train_lists[i]
        batch_ys = train_labels[i]
        # AUGMENT DATA
        # batch_xs = augment_img(batch_xs)
        feeds = {x: batch_xs, y: batch_ys.eval(session=sess), is_training: True}
        sess.run(optm, feed_dict=feeds)
        one_cost = sess.run(cost, feed_dict=feeds)

        total_cnt += 1
        total_cost += one_cost

        # 100步输出一次cost结果
        if total_cnt % 100 == 0:
            print ("total_cnt:%d  cost: %.9f" % (total_cnt, total_cost / total_cnt))

        # 每训练1000张图片，在测试集上测试一下
        if total_cnt % 1000 == 0:
            # 在1000张测试集图片上计算准确度
            val_acc_sum = 0.0
            for j in range(test_photo_cnt):
                test_batch_xs = test_lists[j]
                test_batch_ys = test_labels[j]

                test_feeds = {x: test_batch_xs, y: test_batch_ys.eval(session=sess), is_training: False}

                val_acc = sess.run(accr, feed_dict=test_feeds)
                val_acc_sum = val_acc_sum + val_acc

            val_acc = val_acc_sum / test_photo_cnt

            print (" 在验证数据集上的准确度为: %.5f" % (val_acc))

            # 如果准确率高于之前最好水平，保存模型
            if val_acc > current_best_accuracy:
                current_best_accuracy = val_acc
                savename = savedir + "best_cnt_" + str(total_cnt) + "_accuracy_" + str(current_best_accuracy) + ".ckpt"
                saver.save(sess=sess, save_path=savename)
                print (" [%s] SAVED." % (savename))
    # 循环处理所有训练集一次 end


print ("OPTIMIZATION FINISHED")