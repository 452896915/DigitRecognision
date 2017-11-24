# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from tensorflow.python import debug as tf_debug

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

from tensorflow.examples.tutorials.mnist import input_data

# 数据源
minist = input_data.read_data_sets('./database/minist', one_hot=True)

# 定义各种变量
n_classes = 10  # 分类数目
length = 28  # 图像长度
width = length
height = length

out_frequency = 500  # 每train 500次输出一次cost值
test_frequency = 1000  # 每train 1000次进行一次test

test_photo_batch_cnt = 100  # 测试数据batch数目
test_photo_each_batch_size = 10  # 测试数据每个batch的图片数量

# 定义模型的输入输出
x = tf.placeholder("float", shape=(None, 28 * 28), name="w1")  # 输入的图像28*28
y = tf.placeholder("float", shape=(None, n_classes), name="w2")  # 输出的标签 1*10
is_training = tf.placeholder(tf.bool, name="w3")  # 标志位，是训练还是预测


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def CNN(inputs, is_training=True):
    # 将1*784的输入数据reshape成28*28的ndArray
    shaped_inputs = tf.reshape(inputs, [-1, height, width, 1])  # NHWC  N:Sample的数量 HW:高和宽  C=1 一个通道，灰度值

    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}

    init_func = tf.truncated_normal_initializer(stddev=0.01)  # 正太分布初始化

    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=lrelu,
                        weights_initializer=init_func,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        # 第一个卷积层 16个卷积核
        net = slim.conv2d(shaped_inputs, 16, [5, 5], scope='conv0')

        # 第一个池化层
        net = slim.max_pool2d(net, [2, 2], scope='pool0')

        # 第二个卷积层 32个卷积核
        net = slim.conv2d(net, 32, [5, 5], scope='conv1')
        # 第二个池化层
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # 第三个卷积层 64个卷积核
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        # 第三个池化层
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # 把矩阵flattern成一维的，[batch_size, k]
        net = slim.flatten(net, scope='flatten3')

        # 第一个全连接层
        net = slim.fully_connected(net, 1024,
                                   activation_fn=lrelu,
                                   weights_initializer=init_func,
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params,
                                   scope='fc4')
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training, scope='dr')

        # 第二个全连接层,输出为10个类别
        out = slim.fully_connected(net, n_classes, activation_fn=None, normalizer_fn=None, scope='fco')
        return out


print ("神经网络准备完毕")

# PREDICTION
pred = CNN(x, is_training)
# ppred = tf.nn.softmax(pred) # [N, 10]  softmax归一化

# 预测的时候使用这个节点的值,选10个分类中概率最大的一个作为预测结果
out_result = tf.arg_max(pred, 1, name="op_to_restore")

# LOSS AND OPTIMIZER
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))  # 计算输出和标记结果的交叉熵作为损失函数
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

corr = tf.equal(tf.arg_max(pred, 1), tf.argmax(y, 1))  # 按行取最大值所在的位置，比较预测结果和标注结果是否相同，计算准确率
accr = tf.reduce_mean(tf.cast(corr, "float"))  # 由于一次处理一个batch，一个batch包含多条结果，求多个结果的平均值作为准确度

# INITIALIZER
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print ("FUNCTIONS READY")

    # 存储模型路径
    savedir = "minist_model_out/"
    saver = tf.train.Saver(max_to_keep=100)
    save_step = 4
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print ("SAVER READY")

    # PARAMETERS
    training_epochs = 50  # 在整个训练集上过多少遍
    batch_size = 10  # 每次处理训练集的一个batch包含条目的数量

    val_acc = 0
    val_acc_max = 0
    current_best_accuracy = 0.0

    # OPTIMIZE
    currentTime = time.time()
    total_cost = 0.
    total_cnt = 0
    for epoch in range(training_epochs):  # 循环处理所有训练集多次
        total_batch = int(minist.train.num_examples / batch_size)  # 训练数据集分割成若干个输入batch，一次处理一个batch
        # 循环处理所有训练集一次 start
        for i in range(total_batch):
            batch = minist.train.next_batch(batch_size)  # 一次获取batch_size个元素
            batch_xs = batch[0]  # 对应一条训练数据的748个像素
            batch_ys = batch[1]  # 对应一条训练数据的标注结果

            feeds = {x: batch_xs, y: batch_ys, is_training: True}
            sess.run(optm, feed_dict=feeds)  # 执行一次训练过程
            one_cost = sess.run(cost, feed_dict=feeds)  # 计算本次训练的cost

            total_cnt += 1
            total_cost += one_cost

            # 100步输出一次cost结果
            if total_cnt % out_frequency == 0:
                print ("total_cnt:%d  cost: %.9f" % (total_cnt, total_cost / out_frequency))
                total_cost = 0.

            # 每训练1000次，在测试集上测试一下
            if total_cnt % test_frequency == 0:
                # 在1000张测试集图片上计算准确度
                val_acc_sum = 0.0
                for j in range(test_photo_batch_cnt):
                    test_batch = minist.test.next_batch(test_photo_each_batch_size)
                    test_batch_xs = test_batch[0]
                    test_batch_ys = test_batch[1]

                    test_feeds = {x: test_batch_xs, y: test_batch_ys, is_training: False}

                    val_acc = sess.run(accr, feed_dict=test_feeds)
                    val_acc_sum = val_acc_sum + val_acc

                val_acc = val_acc_sum / test_photo_batch_cnt

                print (" 在验证数据集上的准确度为: %.5f" % (val_acc))

                # 如果准确率高于之前最好水平，保存模型
                if val_acc > current_best_accuracy:
                    current_best_accuracy = val_acc
                    savename = savedir + "best_cnt_" + str(total_cnt) + "_accuracy_" + str(
                        current_best_accuracy) + ".ckpt"
                    saver.save(sess=sess, save_path=savename)
                    print (" [%s] SAVED." % (savename))
                    # 循环处理所有训练集一次 end

    print ("OPTIMIZATION FINISHED")
