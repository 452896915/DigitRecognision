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

mnist=input_data.read_data_sets('./database/minist', one_hot=True)

def predict(val_x, labels):
    feed_dict = {x: val_x, flag: False}

    print "labels: "
    print labels

    print "predicts:"
    print sess.run(op_to_restore, feed_dict)

    val_x.shape = 28, 28 # nparray尺寸由1*784转换成28*28

    plt.imshow(val_x)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

with tf.Session() as sess:
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('./minist_cnn_model_out/best_cnt_1000_accuracy_0.963999994397.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./minist_cnn_model_out/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("w1:0")
    y = graph.get_tensor_by_name("w2:0")
    flag = graph.get_tensor_by_name("w3:0")
    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    for i in range(100):
        batch = mnist.train.next_batch(1)
        batch_xs = batch[0]
        batch_ys = batch[1]

        predict(batch_xs, batch_ys)