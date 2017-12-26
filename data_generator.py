# -*- coding: utf-8 -*-
import numpy as np
import random

from keras.utils import np_utils
from process_wav import *
from config import *


def data_generator(data_lst, batch_size):
    '''
    Generate batch from  data_lst 
    :data_lst:         - list data - [(class_id, user_id, path), ...]
    :data_batch_size:  - batch size
    
    :return:  x_batch, y_batch
                x_batch.shape = (batch_size, frame_count, feature_count)
                y_batch.shape = (batch_size, )
    '''

    # k - sample population length
    k = len(data_lst)
    if k < batch_size:
        raise ValueError('k < batch_size: {} < {} '.format(k, batch_size))
    while True:
        shuffled_data_lst = random.sample(data_lst, k)
       
        for start in range(0, len(shuffled_data_lst), batch_size):
            x_batch = []
            y_batch = []
            end     = min(start + batch_size, len(shuffled_data_lst))
            i_data_batch = shuffled_data_lst[start:end]

            for elem in i_data_batch:
                (class_id, user_id, path) = elem
                x_batch.append(process_wav_file(path))
                y_batch.append(class_id)

            x_batch = np.array(x_batch)
            y_batch = np_utils.to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch

def data_generator_predict(data_lst):
    '''
    Generate feature matrix for data_lst
    :data_lst:         - list data - [(path_to_wav), ...]
   
    :return:  path_to_wav, x_batch
                x_batch.shape = (1, frame_count, feature_count)
    '''

    for item in data_lst:
        yield item,  np.expand_dims(process_wav_file(item), axis=0)


def data_generator_test(data_lst):
    '''
    Generate feature matrix for data_lst
    :data_lst:         - list data - [(class_id, user_id, path), ...]
   
    :return:  class_id, path, x_batch
                x_batch.shape = (1, frame_count, feature_count)
    '''

    for class_id, user_id, path in data_lst:
        yield class_id, path, np.expand_dims(process_wav_file(path), axis=0)




