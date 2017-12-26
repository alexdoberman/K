# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from config import *


def model_dnn(x, input_shape, output_shape):


    _, frame_count, feature_count = input_shape
    _, output_count               = output_shape


    act_fn = tf.nn.relu

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, frame_count*feature_count])

    with tf.variable_scope("fc1"):
        fc1 = tf.layers.dense(x_image, 512, activation=act_fn)

    with tf.variable_scope("fc2"):
        fc2 = tf.layers.dense(fc1, 512, activation=act_fn)

    with tf.variable_scope("fc3"):
        fc3 = tf.layers.dense(fc2, 512, activation=act_fn)

    with tf.variable_scope("output_layer"):
        output_layer = tf.layers.dense(fc3, output_count)

    return output_layer
