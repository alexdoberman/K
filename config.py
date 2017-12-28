# -*- coding: utf-8 -*-

DATADIR = './data' # unzipped train and test data
OUTDIR  = './out'  # just a random name

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

LOG_DIR              =  './log'
VALIDATION_FREQUENCY =  10
BATCH_SIZE           =  64

MAX_STEPS            =  100000
LR                   =  1e-4

RESTORE_MODEL        = False
SAVE_DIR             =  './model/base'
RESTORE_MODEL_NAME   =  'base-99990.meta'


dropout              = 0.4


