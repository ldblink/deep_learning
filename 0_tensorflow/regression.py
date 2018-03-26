# -*- coding: utf-8 -*-
"""
y = x^2 - 0.5
"""

import numpy as np
import tensorflow as tf


# training data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


def add_layer(input, input_size, output_size, activation=None):
    W = tf.Variable(initial_value=tf.random_normal([input_size, output_size]),
                    dtype=tf.float32)
    b = tf.Variable(initial_value=tf.random_normal([1, output_size]),
                    dtype=tf.float32)

    output = tf.matmul(input, W) + b
    if activation:
        output = activation(output)

    return output


xs = tf.placeholder(shape=[None, 1],
                    dtype=tf.float32)
ys = tf.placeholder(shape=[None, 1],
                    dtype=tf.float32)

l1 = add_layer(xs, 1, 10, activation=tf.nn.relu)
l2 = add_layer(l1, 10, 20, activation=tf.nn.tanh)
l3 = add_layer(l2, 20, 10, activation=tf.nn.sigmoid)
l4 = add_layer(l3, 10, 1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - l4), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(20000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


print(sess.run(l4, feed_dict={xs: [[1]]}))
