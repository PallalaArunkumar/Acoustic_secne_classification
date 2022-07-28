import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



import numpy as np
import tensorflow as tf
from tensorflow import keras

#from keras.optimizers import SGD

import sys
sys.path.append("..")
from utils import *
from funcs import *

from fcnn_att import model_fcnn
from DCASE_training_functions import *


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
  
 



# Please put your csv file for train and validation here.
# If you dont generate the extra augmented data, please use 
# ../evaluation_setup/fold1_train.csv and delete the aug_csv part
train_csv = '../evaluation/train1_middle1.csv'
val_csv = '../evaluation/evaluate_middle1.csv'
#aug_csv = 'evaluation_setup/fold1_train_a_2003.csv'

feat_path = '../features/logmel_128'
#aug_path = 'features/logmel128_reverb_scaled/'

experiments = 'exp_fcnn_channelattention_4class/'

if not os.path.exists(experiments):
    os.makedirs(experiments)


#train_aug_csv = generate_train_aug_csv(train_csv, aug_csv, feat_path, aug_path, experiments)
train_aug_csv = train_csv

num_audio_channels = 1
num_freq_bin = 128
num_classes = 4
max_lr = 0.004
batch_size = 30
num_epochs = 300
mixup_alpha = 0.4
crop_length = 400
sample_num = len(open(train_aug_csv, 'r').readlines()) - 1


# compute delta and delta delta for validation data
data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
#print(data_val[:,:,4:-4,:].shape)

data_deltas_val = deltas(data_val)
#print(data_deltas_val[:,:,2:-2,:].shape)
data_deltas_deltas_val = deltas(data_deltas_val)
#print(data_deltas_deltas_val.shape)
data_val = np.concatenate((data_val[:,:,4:-4,:],data_deltas_val[:,:,2:-2,:],data_deltas_deltas_val),axis=-1)
#print(data_val.shape)

y_val = tf.keras.utils.to_categorical(y_val, num_classes)

model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, 3*num_audio_channels], num_filters=[48, 96, 192], wd=0)

model.compile(loss='categorical_crossentropy',
              optimizer =tf.keras.optimizers.SGD(lr=max_lr,decay=1e-6, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.summary()


lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0]) 
save_path = experiments + "/model-{epoch:02d}-{val_accuracy:.4f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks = [lr_scheduler, checkpoint]


# Due to the memory limitation, in the training stage we split the training data
train_data_generator = Generator_timefreqmask_withdelta_splitted(feat_path, train_aug_csv, num_freq_bin,
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length, splitted_num=4)()


history = model.fit(train_data_generator,
                              validation_data=(data_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size = 100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              )
