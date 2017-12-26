# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import wavfile
from mel_spec import *


def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav


def align_signal_to_16000(wav):

    L = 16000  # 1 sec
    
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]
    elif len(wav) < L:
        rem_len = L - len(wav)

        i = np.random.randint(0, rem_len)
        j = rem_len - i 

        eps = 0.000000001
        silence_part_left = np.ones((i,))*eps
        silence_part_right = np.ones((j,))*eps
        wav = np.concatenate([silence_part_left, wav, silence_part_right])

    return wav





def process_wav_file(fname):

    '''
    Get feature matrix for fname wav

    '''

    include_delta = True
    include_acceleration = False
    
    melspec_params = {
        "window" : "hamming_asymmetric",
        "n_mels" : 40,                 # Number of MEL bands used
        "n_fft"  : 512,                 # FFT length
        "win_length" : 512,
        "hop_length" : 256,
        "fmin" : 0,                     # Minimum frequency when constructing MEL bands
        "fmax" : 8000,                  # Maximum frequency when constructing MEL band
        "htk"  : False                  # Switch for HTK-styled MEL-frequency equation
    }

    delta_params = {
        "width": 9
    }

    acceleration_params = {
        "width": 9
    }


    wav_16000 = align_signal_to_16000(read_wav_file(fname))
    feature_matrix = extract_mel_spec(y = wav_16000, fs=16000, include_delta=include_delta,
                        include_acceleration=include_acceleration, melspec_params = melspec_params, 
                        delta_params=delta_params, acceleration_params=acceleration_params)

    return feature_matrix 

