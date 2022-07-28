import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
from scipy.io.wavfile import write  #if gets an error remove this line and type it again


overwrite = True

csv_file = '../../train1_middle.csv'
output_path = '../../newdata/'
#feature_type = 'logmel'
folder_name = "../../newdata/"

sr = 44100
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft/2)
num_time_bin = int(np.ceil(duration*sr/hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep=',',header=None,encoding='ASCII')
wavpath = data_df[0].tolist()


for i in range(len(wavpath)):
    stereo, fs = librosa.load(folder_name + wavpath[i],mono=True,sr=sr)
    noise = np.random.normal(0,1,len(stereo))
    augmented_data = np.where(stereo != 0.0, stereo.astype('float64') + 0.01 * noise, 0.0).astype(np.float32)
    stereo = augmented_data
    filename=wavpath[i].split('.')[0]   #filename
    print("**********\n"+filename)
    
    write(output_path+filename+'_noise.wav',sr,stereo)#saving the file
    
    '''
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

    logmel_data = np.log(logmel_data+1e-8)

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    
    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + wavpath[i][5:-4] + '_noise.' + feature_type
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)'''
        
        

