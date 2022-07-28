import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
from itertools import islice
import random
from scipy.io.wavfile import write


overwrite = True


csv_file = '../../train1_middle.csv'
output_path = '../../newdata/audio'
#feature_type = 'logmel'
folder_name = "../../newdata/audio/"

sr = 44100
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft/2)
num_time_bin = int(np.ceil(duration*sr/hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

data_df = pd.read_csv(csv_file, sep=',',header=None, encoding='ASCII')

wavpath = data_df[0].tolist()
label_dict = dict(home=0, forest_path=1, residential_area=2, transport=3)

def class_sort():
    class_list = []
    for i in range(4):
        ap = []
        class_list.append(ap)
    with open(csv_file, 'r') as csv_r:
       
        for line in islice(csv_r, 0, None):
            file_name = line.split(',')[0].split('/')[1]
            print("*******************\n"+file_name)
            label = line.split(',')[1].split('\n')[0]
            class_list[label_dict[label]].append(file_name)

    return class_list


def data_add(out_dir):
    sample_rate = 44100
    class_list = class_sort()
    for label in class_list:
        length = len(label)
        print(length)
        for file in label:
            if file=='':
               break
            
            y, sr = librosa.load(folder_name + file, mono=True, sr=sample_rate)
            num = random.randint(0, length - 1)
            while file == label[num]:
                num = random.randint(0, length - 1)
            f1, f2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
            #print("#######\n"+folder_name + label[num])
            y2, _ = librosa.load(folder_name + label[num], mono=True, sr=sample_rate)
            stereo = y * f1 + y2 * f2
            filename=file.split('.')[0]   #filename
            write(out_dir+"/"+filename+'_add.wav',sample_rate,stereo)#saving the file
'''
#feature extraction
            logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
            logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=0.0, fmax=sr/2, htk=True, norm=None)

            logmel_data = np.log(logmel_data+1e-8)

            feat_data = logmel_data
            feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    
            feature_data = {'feat_data': feat_data,}

            cur_file_name = output_path + '/' + file.split('.')[0] + '_add.' + feature_type

            pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)'''



if __name__ == "__main__":
    data_add(output_path)
        
        

