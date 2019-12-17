import numpy as np
import librosa
import pandas as pd


def stretch(data, rate=1):
    input_length = 3 * 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


metadata = pd.read_csv('D:\\nlp\\speech_emotion_recognition\\data\\train_label.csv')
for index, row in metadata.iterrows():
    file_name = 'D:/data/esr2019_hcmus/Train1/' + row["File"]
    print(file_name)
    signal, sample_rate = librosa.load(file_name)
    data = stretch(signal)
    np.save('D:/data/esr2019_hcmus/Train2/' + row["File"] + '.npy', data)
