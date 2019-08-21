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
import tensorflow.contrib.rnn as rnn
import time

from tensorflow.examples.tutorials.mnist import input_data

# 数据源
minist = input_data.read_data_sets('./database/minist', one_hot=True)

# 定义各种变量
n_classes = 10  # 分类数目
length = 28  # 图像长度
width = length
height = length

cell_cnt = 200

out_frequency = 500  # 每train 500次输出一次cost值
test_frequency = 1000  # 每train 1000次进行一次test

test_photo_batch_cnt = 100  # 测试数据batch数目
test_photo_each_batch_size = 10  # 测试数据每个batch的图片数量

# 定义模型的输入输出
x = tf.placeholder("float", shape=(None, 28 * 28), name="w1")  # 输入的图像28*28
y = tf.placeholder("float", shape=(None, n_classes), name="w2")  # 输出的标签 1*10

weights = tf.Variable(tf.truncated_normal([cell_cnt, n_classes], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[n_classes]))

def LSTM(X, weights, bias):
    shaped_inputs = tf.reshape(X, [-1, height, width])  # tf.reshape(X, [-1, height, width, 1])
    lstm_cell = rnn.BasicLSTMCell(cell_cnt)  # BasicRNNCell
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, shaped_inputs, dtype="float")  # bidirectional rnn / raw rnn
    results = tf.matmul(state[1], weights) + bias
    return results

print ("神经网络准备完毕")

# PREDICTION
pred = LSTM(x, weights, bias)
# ppred = tf.nn.softmax(pred) # [N, 10]  softmax归一化

# 预测的时候使用这个节点的值,选10个分类中概率最大的一个作为预测结果
out_result = tf.arg_max(tf.nn.softmax(pred), 1, name="op_to_restore")

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

    tf.summary.scalar('cost', cost)
    tf.summary.scalar('accr', accr)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
    test_writer = tf.summary.FileWriter('./summary/test', sess.graph)

    wirter = tf.summary.FileWriter("logs/", sess.graph)

    # 存储模型路径
    savedir = "minist_rnn_model_out/"
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
    total_train_cnt = 0
    total_test_cnt = 0
    for epoch in range(training_epochs):  # 循环处理所有训练集多次
        total_batch = int(minist.train.num_examples / batch_size)  # 训练数据集分割成若干个输入batch，一次处理一个batch
        # 循环处理所有训练集一次 start
        for i in range(total_batch):
            batch = minist.train.next_batch(batch_size)  # 一次获取batch_size个元素
            batch_xs = batch[0]  # 对应一条训练数据的748个像素
            batch_ys = batch[1]  # 对应一条训练数据的标注结果

            feeds = {x: batch_xs, y: batch_ys}
            sess.run(optm, feed_dict=feeds)  # 执行一次训练过程
            summary, one_cost = sess.run([merged, cost], feed_dict=feeds)  # 计算本次训练的cost

            total_train_cnt += 1
            total_cost += one_cost

            # 100步输出一次cost结果
            if total_train_cnt % out_frequency == 0:
                print ("total_cnt:%d  cost: %.9f" % (total_train_cnt, total_cost / out_frequency))
                total_cost = 0.
                train_writer.add_summary(summary, total_train_cnt)

            # 每训练1000次，在测试集上测试一下
            if total_train_cnt % test_frequency == 0:
                # 在1000张测试集图片上计算准确度
                val_acc_sum = 0.0
                for j in range(test_photo_batch_cnt):
                    test_batch = minist.test.next_batch(test_photo_each_batch_size)
                    test_batch_xs = test_batch[0]
                    test_batch_ys = test_batch[1]

                    test_feeds = {x: test_batch_xs, y: test_batch_ys}

                    summary, val_acc = sess.run([merged, accr], feed_dict=test_feeds)
                    val_acc_sum = val_acc_sum + val_acc

                    test_writer.add_summary(summary, total_test_cnt)

                    total_test_cnt = total_test_cnt + 1

                val_acc = val_acc_sum / test_photo_batch_cnt

                print (" 在验证数据集上的准确度为: %.5f" % (val_acc))

                # 如果准确率高于之前最好水平，保存模型
                if val_acc > current_best_accuracy:
                    current_best_accuracy = val_acc
                    savename = savedir + "best_cnt_" + str(total_train_cnt) + "_accuracy_" + str(
                        current_best_accuracy) + ".ckpt"
                    saver.save(sess=sess, save_path=savename)
                    print (" [%s] SAVED." % (savename))
                    # 循环处理所有训练集一次 end

    print ("OPTIMIZATION FINISHED")
