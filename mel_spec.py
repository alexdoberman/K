# -*- coding: utf-8 -*-

import numpy
import librosa
import scipy


def extract_mel_spec(y, fs=16000, include_delta=True,
                       include_acceleration=True, melspec_params=None, delta_params=None, acceleration_params=None):
    """Feature extraction, melspec features

    Outputs features in dict, format:
    
    :feature_matrix: [shape=(frame count, feature vector size)]


    Parameters
    ----------
    y: numpy.array [shape=(signal_length, )]
        Audio

    fs: int > 0 [scalar]
        Sample rate
        (Default value=16000)

    statistics: bool
        Calculate feature statistics for extracted matrix
        (Default value=True)


    include_delta: bool
        Include delta MFCC coefficients.
        (Default value=True)

    include_acceleration: bool
        Include acceleration MFCC coefficients.
        (Default value=True)

    mfcc_params: dict or None
        Parameters for extraction of static MFCC coefficients.

    delta_params: dict or None
        Parameters for extraction of delta MFCC coefficients.

    acceleration_params: dict or None
        Parameters for extraction of acceleration MFCC coefficients.

    Returns
    -------
    result: dict
        Feature dict

    """


    eps = numpy.spacing(1)

    # Windowing function
    if melspec_params['window'] == 'hamming_asymmetric':
        window = scipy.signal.hamming(melspec_params['n_fft'], sym=False)
    elif melspec_params['window'] == 'hamming_symmetric':
        window = scipy.signal.hamming(melspec_params['n_fft'], sym=True)
    elif melspec_params['window'] == 'hann_asymmetric':
        window = scipy.signal.hann(melspec_params['n_fft'], sym=False)
    elif melspec_params['window'] == 'hann_symmetric':
        window = scipy.signal.hann(melspec_params['n_fft'], sym=True)
    else:
        window = None

    # Calculate Static Coefficients
    power_spectrogram = numpy.abs(librosa.stft(y + eps,
                                                   n_fft=melspec_params['n_fft'],
                                                   win_length=melspec_params['win_length'],
                                                   hop_length=melspec_params['hop_length'],
                                                   center=True,
                                                   window=window))**2
    mel_basis = librosa.filters.mel(sr=fs,
                                    n_fft=melspec_params['n_fft'],
                                    n_mels=melspec_params['n_mels'],
                                    fmin=melspec_params['fmin'],
                                    fmax=melspec_params['fmax'],
                                    htk=melspec_params['htk'])
    mel_spectrum = numpy.log(numpy.dot(mel_basis, power_spectrogram) + eps)

    # Collect the feature matrix
    feature_matrix = mel_spectrum
    if include_delta:
        # Delta coefficients
        mel_spectrum_delta = librosa.feature.delta(mel_spectrum, **delta_params)

        # Add Delta Coefficients to feature matrix
        feature_matrix = numpy.vstack((feature_matrix, mel_spectrum_delta))

    if include_acceleration:
        # Acceleration coefficients (aka delta delta)
        mel_spectrum_delta2 = librosa.feature.delta(mel_spectrum, order=2, **acceleration_params)

        # Add Acceleration Coefficients to feature matrix
        feature_matrix = numpy.vstack((feature_matrix, mel_spectrum_delta2))


    feature_matrix = feature_matrix.T

    return feature_matrix
