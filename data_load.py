# -*- coding: utf-8 -*-

import os
import re
from glob import glob
import numpy as np
from scipy.io import wavfile
from config import *

def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(data_dir +'/'+ 'train/audio/*/*wav')
    
    # Fix windows path style
    for i in range(len(all_files)):
        all_files[i] = all_files[i].replace('\\', '/')

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val

def load_data_predict(data_dir):

    all_files = glob(data_dir +'/'+ 'test/audio/*wav')
    
    # Fix windows path style
    for i in range(len(all_files)):
        all_files[i] = all_files[i].replace('\\', '/')

    return all_files


def load_data_test(data_dir):
    """ Return 1 lists of tuples:
    [(class_id, user_id, path), ...] for test
    """

    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")

    with open(os.path.join(data_dir, 'train/testing_list.txt'), 'r') as fin:
        test_files = fin.readlines()

    possible = set(POSSIBLE_LABELS)
    test = []
    audio_path = data_dir + '/train/audio/'

    for entry in test_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)

            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, audio_path +  entry.rstrip())
            test.append(sample)

    print('There are {} test samples'.format(len(test)))
    return test



def data_generator(data, params, mode='train'):
    def generator():
        if mode == 'train':
            np.random.shuffle(data)
        # Feel free to add any augmentation
        for (label_id, uid, fname) in data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0
                    yield dict(
                        target=np.int32(label_id),
                        wav=wav[beg: beg + L],
                    )
            except Exception as err:
                print(err, label_id, uid, fname)
    return generator

