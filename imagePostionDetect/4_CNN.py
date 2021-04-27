import os

import tensorflow as tf
import random
import cv2
import numpy as np
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def walk_files(path='.', endpoint='.png'):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(endpoint):
                file_list.append(file_path)
    return file_list


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, ys: v_ys, keep_prob: keep_p})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # print(sess.run(tf.argmax(v_ys, 1), feed_dict={xs: v_xs, ys: v_ys, keep_prob: 0.5}))
    # print(sess.run(correct_prediction, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 0.5}))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def get_feature_and_label(path):
    feature = []
    label = []
    img_list = walk_files(path, ".png")
    for _ in range(100):
        file_name = img_list[int(random.random() * len(img_list))]
        img = cv2.imread(file_name, 0)
        img = cv2.warpAffine(img, np.float32([[28.0 / img.shape[1], 0, 0], [0, 28.0 / img.shape[0], 0]]), (28, 28)) / 256.0
        feature.extend(img.reshape(1, -1).tolist())
        l = [0, 0, 0]
        l[int(file_name[-5])] = 1
        label.extend(l)
    return np.float32(feature).reshape(100, -1), np.float32(label).reshape(100, -1)


if __name__ == "__main__":

    pic_row = 28
    pic_col = 28
    keep_p = 0.5
    class_size = 3
    xs = tf.placeholder(tf.float32, [None, pic_row * pic_col], name='xs_holder')
    ys = tf.placeholder(tf.float32, [None, class_size], name='ys_holder')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    x_image = tf.reshape(xs, [-1, pic_row, pic_col, 1], name='x_image')

    # conv layer 1
    W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')  # patch size 5x5, in size: 1, out size: 32
    b_conv1 = bias_variable([32], name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='h_conv1')  # 28x28x32
    h_pool1 = max_pool_2x2(h_conv1, name='h_pool1')  # 14x14x32

    # conv layer 2
    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # patch size 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64], name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')  # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2, name='h_pool2')  # output size 7x7x64

    W_fc1 = weight_variable([int(pic_row / 4) * int(pic_col / 4) * 64, 1024], name='W_fc1')
    b_fc1 = bias_variable([1024], name='b_fc1')

    h_pool2_flat = tf.reshape(h_pool2, [-1, int(pic_row / 4) * int(pic_col / 4) * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    W_fc2 = weight_variable([1024, class_size], name='W_fc2')
    b_fc2 = bias_variable([class_size], name='b_fc2')
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='prediction')

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]), name='entropy')
    train_step = tf.train.AdamOptimizer(1e-4, name='train_step').minimize(cross_entropy)

    sess = tf.Session()
    # writer = tf.summary.FileWriter("./logs")
    # writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    # print(tf.global_variables())

    for epoch in range(10000000):
        test_xs, test_ys = get_feature_and_label("./test")
        train_xs, train_ys = get_feature_and_label("./train")

        sess.run(train_step, feed_dict={xs: train_xs, ys: train_ys, keep_prob: keep_p})
        if epoch % 5 == 0:
            print('*', epoch)
            print('acc in train data', compute_accuracy(train_xs, train_ys))
            acc = compute_accuracy(test_xs, test_ys)
            print('acc in test  data', acc)
        if epoch % 50 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './CNN_MODEL/my_model')
