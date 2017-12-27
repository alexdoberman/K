# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import config
from data_load import *
from data_generator import *
from tf_model import *



def loss_func (y, y_):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    return cross_entropy

def accuracy_func(y, y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def  main():

    ############################################################
    # Import data
    trainset, valset = load_data(config.DATADIR)
    train_gen        = data_generator(trainset, batch_size = config.BATCH_SIZE)
    val_gen          = data_generator(valset, batch_size = config.BATCH_SIZE)
    ############################################################


    ############################################################
    # Get input/output sizes
    probe_x_batch, probe_y_batch =  next(train_gen)
    (batch_size, frame_count, feature_count) = probe_x_batch.shape
    output_count = len(config.POSSIBLE_LABELS)

    input_shape  = (None, frame_count, feature_count)
    output_shape = (None, output_count)


    print ('--------------------------------------------------')
    print ('Train params:')
    print ('batch_size = {} frame_count = {} feature_count = {} output_count = {}'.format(batch_size, frame_count, feature_count, output_count))
    print ('input_shape  = {}'.format(input_shape))
    print ('output_shape = {}'.format(output_shape))
    print ('--------------------------------------------------')
    ############################################################


    ############################################################
    # Input placeholders
    with tf.name_scope('input'):
        x  = tf.placeholder(tf.float32, [None, frame_count, feature_count], name = "x")
        y_ = tf.placeholder(tf.float32, [None, output_count], name = "y_")
        keep_prob = tf.placeholder(tf.float32, name = "keep_prob") 
    

    # Model define
    with tf.name_scope('model_deepnn'):
        y = get_model(x, input_shape, output_shape, keep_prob)
        y = tf.identity(y, "y")

    # Accuracy
    with tf.name_scope('accuracy'):        
        acc = accuracy_func(y, y_)
        tf.summary.scalar('accuracy', acc)
    
    # Loss
    with tf.name_scope('loss'):
        loss = loss_func (y, y_)
        tf.summary.scalar('loss', loss)

    # Optimizer
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(config.LR).minimize(loss)

    merged = tf.summary.merge_all()    
    saver  = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(config.LOG_DIR + '/train', sess.graph)
        test_writer  = tf.summary.FileWriter(config.LOG_DIR + '/test', sess.graph)

        for step in range(config.MAX_STEPS):
            x_batch, y_batch =  next(train_gen)

            if step % config.VALIDATION_FREQUENCY == 0:
                x_batch_val, y_batch_val =  next(val_gen)

                summary, accuracy = sess.run([merged, acc], feed_dict={x: x_batch_val, 
                                                                       y_: y_batch_val, 
                                                                       keep_prob: 1.0})
                test_writer.add_summary(summary, step)
                print('Accuracy at step %s: %s' % (step, accuracy))

                # Save the graph
                saver.save(sess, config.SAVE_DIR, global_step = step)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict={x: x_batch, 
                                                                       y_: y_batch,
                                                                       keep_prob: config.dropout})
                train_writer.add_summary(summary, step)





'''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(config.LOG_DIR + '/train', sess.graph)
        test_writer  = tf.summary.FileWriter(config.LOG_DIR + '/test')

        for step in range(config.MAX_STEPS):
            x_batch, y_batch =  next(train_gen)

            if step % config.VALIDATION_FREQUENCY == 0:
                x_batch_val, y_batch_val =  next(val_gen)

                summary, accuracy = sess.run([merged, acc], feed_dict={x: x_batch_val, y_: y_batch_val})
                test_writer.add_summary(summary, step)
                print('Accuracy at step %s: %s' % (step, accuracy))

                # Save the graph
                saver.save(sess, config.SAVE_DIR, global_step = step)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict={x: x_batch, y_: y_batch})
                train_writer.add_summary(summary, step)

'''

if __name__ == '__main__':
    print ('tf.__version__ = ', tf.__version__)
    main()
    
    
    

