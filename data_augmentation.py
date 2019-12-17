import numpy as np
import scipy.io.wavfile
import librosa
import pandas as pd
import random
import argparse
import cv2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Audio Augmentation')

    parser.add_argument('--path', dest='path', help='Directory path containing data',
                        default='D:/data/esr2019_hcmus/Train1/', type=str)
    parser.add_argument('--target', dest='target', help='Target directory path to save new data',
                        default='D:/data/esr2019_hcmus/Train1/', type=str)
    parser.add_argument('--csv', dest='csv', help='CSV path',
                        default='D:\\nlp\\speech_emotion_recognition\\data\\train_label.csv', type=str)
    parser.add_argument('--new_csv', dest='new_csv', help='New CSV path',
                        default='D:/data/esr2019_hcmus/train_label1.csv', type=str)
    args = parser.parse_args()

    return args


def augment_roll(wav, shift):
    """
    Rotate a waveform along time-axis.
    :param wav: a waveform.
    :param shift: shift length.
    :return: a rolled waveform.
    """
    return np.roll(wav, shift)


def augment_stretch(wav, rate=1, length=None):
    """
    Stretch a waveform along time-axis.
    :param wav: a waveform.
    :param rate: stretch rate.
    :param length: fixed length of the stretched waveform. optional.
    :return: a stretched waveform.
    """
    wav = librosa.effects.time_stretch(wav, rate)
    if length:
        if len(wav) > length:
            wav = wav[:length]
        else:
            wav = np.pad(wav, (0, max(0, length - len(wav))), "constant")
    return wav


def augment_pitch(wav, sr, n_steps_lower=-3., n_steps_upper=3.):
    """
    Heighten or lower a waveform's pitch by a randomly selected half-steps.
    :param wav: a waveform.
    :param sr: sample rate.
    :param n_steps_lower: lower bound of the number of half-steps.
    :param n_steps_upper: upper bound of the number of half-steps.
    :return: a increased or decreased waveform.
    """
    n_steps = random.uniform(n_steps_lower, n_steps_upper)
    wav = librosa.effects.pitch_shift(wav, sr, n_steps)
    return wav


def speech_tuning(wav):
    input_length = 16000 * 3
    speed_rate = np.random.uniform(0.7, 1.3)
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
    print('speed rate: %.3f' % speed_rate, '(lower is faster)')
    if len(wav_speed_tune) < input_length:
        pad_len = input_length - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    else:
        cut_len = len(wav_speed_tune) - input_length
        wav_speed_tune = wav_speed_tune[int(cut_len / 2):int(cut_len / 2) + input_length]
    return wav_speed_tune


def main():
    args = parse_args()

    data_path = args.path
    data_des = args.target
    metadata = pd.read_csv(args.csv)
    list_file = []

    for index, row in metadata.iterrows():
        file_name = data_path + row["File"]
        name_file = row["File"].split('.')[0]
        data, sr = librosa.core.load(file_name)
        list_file.append([row["File"], row["Label"]])
        np.save(data_des + row["File"] + '.npy', data)

        #
        # Augment Pitch
        data_pitch = augment_pitch(data, sr)
        np.save(data_des + name_file + '_pitch.wav' + '.npy', data_pitch)
        list_file.append([name_file + '_pitch.wav' + '.npy', row["Label"]])

        # Speech Turning
        data = speech_tuning(data)
        np.save(data_des + name_file + '_speech_turn.wav' + '.npy', data)
        list_file.append([name_file + '_speech_turn.wav' + '.npy', row["Label"]])

        # Augment Roll
        data_roll = augment_roll(data, shift=2)
        np.save(data_des + name_file + '_roll.wav' + '.npy', data_roll)
        list_file.append([name_file + '_roll.wav' + '.npy', row["Label"]])

        print(row["File"])

    print('Finished data augmentation from ', len(list_file), ' files')

    df = pd.DataFrame(list_file, columns=['File', 'Label'])
    df.to_csv(args.new_csv, index=None, header=True)


if __name__ == '__main__':
    main()
