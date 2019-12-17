# %matplotlib inline
import librosa
from scipy.fftpack import dct
import numpy as np
import argparse
import pandas as pd
import os


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Audio Augmentation')

    parser.add_argument('--path', dest='path', help='Directory path containing data',
                        default='D:/data/esr2019_hcmus/Train1/', type=str)
    parser.add_argument('--target', dest='target', help='Target path containing data',
                        default='D:/data/esr2019_hcmus/mfcc/', type=str)
    parser.add_argument('--csv', dest='csv', help='CSV path',
                        default='', type=str)
    parser.add_argument('--frame_size', dest='frame_size', help='Frame size',
                        default=0.025, type=float)
    parser.add_argument('--frame_stride', dest='frame_stride', help='Frame stride',
                        default=0.01, type=float)
    parser.add_argument('--num_ceps', dest='num_ceps', help='Number of Cepstral',
                        default=20, type=int)
    parser.add_argument('--cep_lifter', dest='cep_lifter',
                        help='Refers to the dimensionality of the MFCC vector in the original formulation.',
                        default=22, type=int)
    args = parser.parse_args()

    return args


def get_mfcc(signal, sample_rate, num_ceps=20, cep_lifter=22, frame_size=0.025, frame_stride=0.01):
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])  # Perfect
    # Framing
    # Split the signal into short-time frames

    # Convert from seconds to samples
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate

    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step,
                                                                                       frame_step),
                                                                             (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Window
    frames *= np.hamming(frame_length)
    # Fourier-Transform and Power Spectrum
    NFFT = 512  # or 256
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    # Filter Banks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc


def norm(a):
    return (a - np.mean(a)) / np.std(a)


def stretch(data, rate=1):
    input_length = 3 * 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def main():
    args = parse_args()
    pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    if args.csv != '':
        metadata = pd.read_csv(args.csv)

        for index, row in metadata.iterrows():
            file_name = args.path + row["File"]
            print(file_name)
            signal, sample_rate = librosa.load(file_name)
            mfcc = get_mfcc(signal, sample_rate, args.num_ceps, args.cep_lifter, args.frame_size, args.frame_stride)
            padded_mfcc = pad2d(mfcc.T, 20)
            norm_mfcc = norm(padded_mfcc)
            np.save(args.target + row["File"] + '.npy', norm_mfcc)
    else:
        listfile = os.listdir(args.path)
        for i, f in enumerate(listfile):
            file_name = args.path + f
            print(file_name)
            signal, sample_rate = librosa.load(file_name)
            # mfcc = get_mfcc(signal, sample_rate, args.num_ceps, args.cep_lifter, args.frame_size, args.frame_stride)
            # padded_mfcc = pad2d(mfcc.T, 20)
            # norm_mfcc = norm(padded_mfcc)
            data = stretch(signal)
            np.save(args.target + f + '.npy', data)


if __name__ == '__main__':
    main()
