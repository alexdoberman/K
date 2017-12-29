# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from config import *

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

######################################################
# Factory function
def get_model(x, input_shape, output_shape, keep_prob):
    return model_cnn_v2(x, input_shape, output_shape, keep_prob)


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




def model_cnn_v1(x, input_shape, output_shape, keep_prob):

    _, frame_count, feature_count = input_shape
    _, output_count               = output_shape


    ################################################################

    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1, frame_count, feature_count, 1])

    with tf.variable_scope("conv_1"):
        x = tf.layers.conv2d(x, filters = 32, kernel_size = (5, 5), padding='valid', activation = tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=2)

    with tf.variable_scope("conv_2"):
        x = tf.layers.conv2d(x, filters = 16, kernel_size = (5, 5), padding='valid', activation = tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=2)

    with tf.variable_scope("dropout_1"):
        x = tf.nn.dropout(x, keep_prob)

    with tf.variable_scope("fc_1"):
#        x = tf.layers.flatten(x)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units = 512, activation = tf.nn.relu)

    with tf.variable_scope("dropout_2"):
        x = tf.nn.dropout(x, keep_prob)

    with tf.variable_scope("output_layer"):
        output_layer = tf.layers.dense(x, output_count)

    return output_layer

def model_cnn_v2(x, input_shape, output_shape, keep_prob):

    _, frame_count, feature_count = input_shape
    _, output_count               = output_shape


    ################################################################

    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1, frame_count, feature_count, 1])

    with tf.variable_scope("conv_1"):
        x = tf.layers.conv2d(x, filters = 64, kernel_size = (5, 5), padding='valid', activation = tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=2)

    with tf.variable_scope("conv_2"):
        x = tf.layers.conv2d(x, filters = 32, kernel_size = (5, 5), padding='valid', activation = tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=2)

    with tf.variable_scope("conv_3"):
        x = tf.layers.conv2d(x, filters = 16, kernel_size = (5, 5), padding='valid', activation = tf.nn.relu)
        x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=2)

    with tf.variable_scope("dropout_1"):
        x = tf.nn.dropout(x, keep_prob)

    with tf.variable_scope("fc_1"):
#        x = tf.layers.flatten(x)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units = 512, activation = tf.nn.relu)

    with tf.variable_scope("dropout_2"):
        x = tf.nn.dropout(x, keep_prob)

    with tf.variable_scope("output_layer"):
        output_layer = tf.layers.dense(x, output_count)

    return output_layer



