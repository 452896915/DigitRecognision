# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('database/format2/train_32x32.mat')
print mat.keys()

vals = mat.get("X")
labels = mat.get("y")


for i in range(np.shape(vals)[3]): # 遍历73257张图片中的每一张
    red_vals = vals[:, :, 0, i]
    green_vals = vals[:, :, 1, i]
    blue_vals = vals[:, :, 2, i]
    label = labels[i]

    print label

    w = np.shape(vals)[0]
    h = np.shape(vals)[1]
    gray_vals = [[0 for x in range(w)] for y in range(h)]
    for x in range(w):
        for y in range(h):
            gray_vals[x][y] = (red_vals[x][y] * 30 + green_vals[x][y] * 59 + blue_vals[x][y] * 11 + 50) / 100

    plt.imshow(gray_vals, cmap='Greys_r')
    plt.axis('off')
    plt.show()


