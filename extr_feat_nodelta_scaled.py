import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool



file_path = 'newdata/'
csv_file = 'evaluation/train1_middle1.csv'
output_path = 'features/logmel_128/'
feature_type = 'logmel'

sr = 44100
duration_sec = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration_sec * sr / hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep=',', encoding='ASCII')
wavpath = data_df['filename'].tolist()
#print("***********\n",wavpath)

for i in range(len(wavpath)):
    stereo, fs = librosa.load(file_path + wavpath[i],mono=True,sr=sr, duration=10)       
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    #print(logmel_data.shape)
    
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)
    

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + wavpath[i][5:-3] + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
     
        

