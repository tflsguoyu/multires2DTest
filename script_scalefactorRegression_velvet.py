## readme
# go to anaconda prompt:
#conda create --name tensorflow-gpu python=3.5
#activate tensorflow-gpu
#conda install jupyter (optional)
#conda install scipy
#pip install tensorflow
#pip install keras
#conda install spyder (optional)
#pip install Pillow
#conda install h5py

# test
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#
#
#datagen = ImageDataGenerator(
#        rotation_range=40,
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True,
#        fill_mode='nearest')
#
#img = load_img('../../ICS274c_data/project/data/train/cats/cat.1.jpg')  # this is a PIL image
#x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
## the .flow() command below generates batches of randomly transformed images
## and saves the results to the `preview/` directory
#i = 0
#for batch in datagen.flow(x, batch_size=1,
#                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#    i += 1
#    if i > 20:
#        break  # otherwise the generator would loop indefinitely
import os
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from scipy.misc import imread

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.layers import Merge
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Concatenate
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization

def load_data_from_npy(image_dir, value_dir, N):
    
    volumes = np.zeros((N,32,32,1))
    for iter in range(N):
        volume = imread(image_dir + 'im%06d.png' % iter, mode='F')/255
        volumes[iter,:,:,0] = volume               
               
    info = np.loadtxt(value_dir, delimiter=',');         
    X = info[:,-6:-3]
    X[:,1] = np.log10(X[:,1])
    Y = info[:,-1]
    Y = (1 / (1 - 0.95 * info[:,-1])) / 20
#    print(Y)
        
#    print(volumes.shape)
#    print(X.shape)
#    print(Y.shape)

    return volumes, X, Y

def learning_rate_schedule(epoch):
    if epoch <= 50:
        return 0.001
    else:
        return 0.0001

class VoxNet(object):
    def __init__(self):
        self.optimizer = keras.optimizers.Adam(lr=0.001)

        self.lr_schedule = LearningRateScheduler(learning_rate_schedule)

        self.histories = []

        density_input = Input(shape=(32, 32, 1), name='density_input')
        x = Convolution2D(filters=32,
                          kernel_size=(5, 5),
                          strides=(2, 2),
                          data_format='channels_last',
                          activation='relu',
                          kernel_regularizer=l2(0.0001),
                          bias_regularizer=l2(0.0001))(density_input)
        print("Layer 1: Conv2D shape={0}".format(x.shape))
        x = Convolution2D(filters=32,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          data_format='channels_last',
                          activation='relu',
                          kernel_regularizer=l2(0.0001),
                          bias_regularizer=l2(0.0001))(x)
        print("Layer 2: Conv2D shape={0}".format(x.shape))
        x = MaxPooling2D(pool_size=(2, 2),
                         strides=None,
                         data_format='channels_last')(x)
        print("Layer 3: MaxPool2D shape={0}".format(x.shape))
        x = Flatten()(x)
        print("Layer 4: Flatten shape={0}".format(x.shape))
        density_out = Dense(units=256,
                  activation='relu',
                  kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001))(x)
        density_out = BatchNormalization()(density_out)
        print("Layer 5: Dense shape={0}".format(density_out.shape))

        feature_input = Input(shape=(3,), name='feature_input')
        feature_out = Dense(units=20,
                  activation='relu',
                  kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001))(feature_input)
        feature_out = BatchNormalization()(feature_out)
        print("Layer 6: Dense shape={0}".format(feature_out.shape))

        x = keras.layers.concatenate([density_out, feature_out])
        print("Layer 7: Merge shape={0}".format(x.shape))
        x = Dense(units=276,
                  activation='relu',
                  kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001))(x)
        print("Layer 8: Dense shape={0}".format(x.shape))
        scale_output = Dense(units=1,
                             activation='sigmoid',
                             kernel_regularizer=l2(0.0001),
                             bias_regularizer=l2(0.0001))(x)
        print("Layer 9: Dense shape={0}".format(scale_output.shape))

        self.model = Model(inputs=[density_input, feature_input], outputs=scale_output)
        self.model.compile(loss='mean_absolute_error', optimizer=self.optimizer)
        print("Model compiled!")

    def load_weights(self, filename):
        print("Load model from {0}".format(filename))
        self.model.load_weights(filename)

    def save_weights(self):
        time_now = datetime.datetime.now()
        time_now = "_{0}_{1}_{2}_{3}_{4}_{5}".format(time_now.year, time_now.month, time_now.day,
                                                     time_now.hour, time_now.minute, time_now.second)
        weights_dir = "weights/"
        weights_filename = weights_dir + "weights" + time_now + ".h5"
        print("Save model at {0}".format(weights_filename))
        self.model.save_weights(weights_filename, False)
        return weights_filename

    def fit(self, volumes_train, X_train, Y_train, epochs):
        print("Start training.")
        history = self.model.fit([volumes_train, X_train],
                                 Y_train,
                                 epochs=epochs,
                                 validation_split=0.1,
                                 batch_size=32,
                                 verbose=2,
                                 callbacks=[self.lr_schedule])
        self.histories = [history]
        return self.save_weights(), history

    def predict(self, volumes_in, X_in):
        return self.model.predict([volumes_in, X_in])

    def evaluate(self, volumes_test, X_test, Y_test):
        self.score = self.model.evaluate([volumes_test, X_test], Y_test,
                                         verbose=0)
        print("Test score:", self.score)

    def continue_fit(self, weights_filename,
                     volumes_train, X_train, Y_train, epochs):
        print("Continue training.")
        self.load_weights(weights_filename)
        history = self.model.fit([volumes_train, X_train],
                                 Y_train,
                                 epochs=epochs,
                                 validation_split=0.1,
                                 batch_size=32,
                                 verbose=2,
                                 callbacks=[self.lr_schedule])
        self.histories.append(history)
        return self.save_weights(), history
    
    
#voxnet = VoxNet()

# In[]
voxnet = VoxNet()

# In[] training

#velvet_0, velvetX_0, velvetY_0 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_0/data.csv', 608)
#velvet_05, velvetX_05, velvetY_05 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_0.5/data.csv', 608)
#velvet_1, velvetX_1, velvetY_1 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_1/data.csv', 608)
#velvet_15, velvetX_15, velvetY_15 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_1.5/data.csv', 608)
#velvet_2, velvetX_2, velvetY_2 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_2/data.csv', 608)
#velvet_25, velvetX_25, velvetY_25 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_2.5/data.csv', 608)
#velvet_3, velvetX_3, velvetY_3 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_3/data.csv', 608)
#velvet_35, velvetX_35, velvetY_35 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_3.5/data.csv', 608)
#velvet_4, velvetX_4, velvetY_4 = load_data_from_npy('input/velvet/velvet_deeplearning/', 'output/velvet_0.95_100_down04_4/data.csv', 608)
#
#
#gabardine_0, gabardineX_0, gabardineY_0 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_0/data.csv', 273)
#gabardine_05, gabardineX_05, gabardineY_05 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_0.5/data.csv', 273)
#gabardine_1, gabardineX_1, gabardineY_1 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_1/data.csv', 273)
#gabardine_15, gabardineX_15, gabardineY_15 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_1.5/data.csv', 273)
#gabardine_2, gabardineX_2, gabardineY_2 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_2/data.csv', 273)
#gabardine_25, gabardineX_25, gabardineY_25 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_2.5/data.csv', 273)
#gabardine_3, gabardineX_3, gabardineY_3 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_3/data.csv', 273)
#gabardine_35, gabardineX_35, gabardineY_35 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_3.5/data.csv', 273)
#gabardine_4, gabardineX_4, gabardineY_4 = load_data_from_npy('input/gabardine/gabardine_deeplearning/', 'output/gabardine_0.95_100_down04_4/data.csv', 273)
#
#
#felt_0, feltX_0, feltY_0 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_0/data.csv', 609)
#felt_05, feltX_05, feltY_05 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_0.5/data.csv', 609)
#felt_1, feltX_1, feltY_1 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_1/data.csv', 609)
#felt_15, feltX_15, feltY_15 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_1.5/data.csv', 609)
#felt_2, feltX_2, feltY_2 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_2/data.csv', 609)
#felt_25, feltX_25, feltY_25 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_2.5/data.csv', 609)
#felt_3, feltX_3, feltY_3 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_3/data.csv', 609)
#felt_35, feltX_35, feltY_35 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_3.5/data.csv', 609)
#felt_4, feltX_4, feltY_4 = load_data_from_npy('input/felt/felt_deeplearning/', 'output/felt_0.95_100_down04_4/data.csv', 609)
#
#volumes_train = np.concatenate((velvet_0,velvet_05,velvet_1,velvet_15,velvet_2,velvet_25,velvet_3,velvet_35,velvet_4,\
#                                gabardine_0,gabardine_05,gabardine_1,gabardine_15,gabardine_2,gabardine_25,gabardine_3,gabardine_35,gabardine_4,\
#                                felt_0,felt_05,felt_1,felt_15,felt_2,felt_25,felt_3,felt_35,felt_4))
#X_train = np.concatenate((velvetX_0,velvetX_05,velvetX_1,velvetX_15,velvetX_2,velvetX_25,velvetX_3,velvetX_35,velvetX_4,\
#                          gabardineX_0,gabardineX_05,gabardineX_1,gabardineX_15,gabardineX_2,gabardineX_25,gabardineX_3,gabardineX_35,gabardineX_4,\
#                          feltX_0,feltX_05,feltX_1,feltX_15,feltX_2,feltX_25,feltX_3,feltX_35,feltX_4))
#Y_train = np.concatenate((velvetY_0,velvetY_05,velvetY_1,velvetY_15,velvetY_2,velvetY_25,velvetY_3,velvetY_35,velvetY_4,\
#                          gabardineY_0,gabardineY_05,gabardineY_1,gabardineY_15,gabardineY_2,gabardineY_25,gabardineY_3,gabardineY_35,gabardineY_4,\
#                          feltY_0,feltY_05,feltY_1,feltY_15,feltY_2,feltY_25,feltY_3,feltY_35,feltY_4))
#
##volumes_train = np.concatenate((velvet_0,velvet_05,velvet_1,velvet_15,velvet_2,velvet_25,velvet_3,velvet_35,velvet_4))
##X_train = np.concatenate((velvetX_0,velvetX_05,velvetX_1,velvetX_15,velvetX_2,velvetX_25,velvetX_3,velvetX_35,velvetX_4))
##Y_train = np.concatenate((velvetY_0,velvetY_05,velvetY_1,velvetY_15,velvetY_2,velvetY_25,velvetY_3,velvetY_35,velvetY_4))
#
##volumes_train = np.concatenate((gabardine_0,gabardine_05,gabardine_1,gabardine_15,gabardine_2,gabardine_25,gabardine_3,gabardine_35,gabardine_4))
##X_train = np.concatenate((gabardineX_0,gabardineX_05,gabardineX_1,gabardineX_15,gabardineX_2,gabardineX_25,gabardineX_3,gabardineX_35,gabardineX_4))
##Y_train = np.concatenate((gabardineY_0,gabardineY_05,gabardineY_1,gabardineY_15,gabardineY_2,gabardineY_25,gabardineY_3,gabardineY_35,gabardineY_4))
#
##volumes_train = np.concatenate((felt_0,felt_05,felt_1,felt_15,felt_2,felt_25,felt_3,felt_35,felt_4))
##X_train = np.concatenate((feltX_0,feltX_05,feltX_1,feltX_15,feltX_2,feltX_25,feltX_3,feltX_35,feltX_4))
##Y_train = np.concatenate((feltY_0,feltY_05,feltY_1,feltY_15,feltY_2,feltY_25,feltY_3,feltY_35,feltY_4))
#
##print(volumes_train.shape)
##print(X_train.shape)
##print(Y_train.shape)
#
#weights_filename, training_history = voxnet.fit(volumes_train, X_train, Y_train, epochs=60)

# In[] predicting

voxnet.load_weights('weights/weights_2017_6_2_20_17_53_allThree.h5')

for i in range(9):
    imageFolder = 'input/velvet/velvet' + repr(i+1) + '/'
    paraFolder = 'output/velvet' + repr(i+1) + '_0.95_100_down04/'
    volumes_valid, X_valid, Y_valid = load_data_from_npy(imageFolder, paraFolder + 'data.csv', 15*(i+1))
    Y_pred = voxnet.predict(volumes_valid, X_valid)
    Y_pred = (1-1/(20*Y_pred))/0.95
             
    with open(paraFolder + 'predict.csv','wb') as fd:    
        np.savetxt(fd, Y_pred, delimiter=',')
    print("Save predict results to {0}".format(paraFolder + 'predict.csv'))

for i in range(4):
    imageFolder = 'input/gabardine/gabardine' + repr(i+1) + '/'
    paraFolder = 'output/gabardine' + repr(i+1) + '_0.95_100_down04/'
    volumes_valid, X_valid, Y_valid = load_data_from_npy(imageFolder, paraFolder + 'data.csv', 20*(i+1))
    Y_pred = voxnet.predict(volumes_valid, X_valid)
    Y_pred = (1-1/(20*Y_pred))/0.95
             
    with open(paraFolder + 'predict.csv','wb') as fd:    
        np.savetxt(fd, Y_pred, delimiter=',')
    print("Save predict results to {0}".format(paraFolder + 'predict.csv'))

for i in range(11):
    imageFolder = 'input/felt/felt' + repr(i+1) + '/'
    paraFolder = 'output/felt' + repr(i+1) + '_0.95_100_down04/'
    volumes_valid, X_valid, Y_valid = load_data_from_npy(imageFolder, paraFolder + 'data.csv', 15*(i+1))
    Y_pred = voxnet.predict(volumes_valid, X_valid)
    Y_pred = (1-1/(20*Y_pred))/0.95
             
    with open(paraFolder + 'predict.csv','wb') as fd:    
        np.savetxt(fd, Y_pred, delimiter=',')
    print("Save predict results to {0}".format(paraFolder + 'predict.csv'))

# In[]:
#volumes_velvet1, X_velvet1, Y_velvet1 = load_data_from_npy('input/velvet/velvet1/', 'output/velvet1_0.95_100_down04/data.csv', 15)
#volumes_velvet2, X_velvet2, Y_velvet2 = load_data_from_npy('input/velvet/velvet2/', 'output/velvet2_0.95_100_down04/data.csv', 30)
#volumes_velvet3, X_velvet3, Y_velvet3 = load_data_from_npy('input/velvet/velvet3/', 'output/velvet3_0.95_100_down04/data.csv', 45)
#volumes_velvet4, X_velvet4, Y_velvet4 = load_data_from_npy('input/velvet/velvet4/', 'output/velvet4_0.95_100_down04/data.csv', 60)
#volumes_velvet5, X_velvet5, Y_velvet5 = load_data_from_npy('input/velvet/velvet5/', 'output/velvet5_0.95_100_down04/data.csv', 75)
#volumes_velvet6, X_velvet6, Y_velvet6 = load_data_from_npy('input/velvet/velvet6/', 'output/velvet6_0.95_100_down04/data.csv', 90)
#volumes_velvet7, X_velvet7, Y_velvet7 = load_data_from_npy('input/velvet/velvet7/', 'output/velvet7_0.95_100_down04/data.csv', 105)
#volumes_velvet8, X_velvet8, Y_velvet8 = load_data_from_npy('input/velvet/velvet8/', 'output/velvet8_0.95_100_down04/data.csv', 120)
#volumes_velvet9, X_velvet9, Y_velvet9 = load_data_from_npy('input/velvet/velvet9/', 'output/velvet9_0.95_100_down04/data.csv', 135)
#
#volumes_gabardine1, X_gabardine1, Y_gabardine1 = load_data_from_npy('input/gabardine/gabardine1/', 'output/gabardine1_0.95_100_down04/data.csv', 20)
#volumes_gabardine2, X_gabardine2, Y_gabardine2 = load_data_from_npy('input/gabardine/gabardine2/', 'output/gabardine2_0.95_100_down04/data.csv', 40)
#volumes_gabardine3, X_gabardine3, Y_gabardine3 = load_data_from_npy('input/gabardine/gabardine3/', 'output/gabardine3_0.95_100_down04/data.csv', 60)
#volumes_gabardine4, X_gabardine4, Y_gabardine4 = load_data_from_npy('input/gabardine/gabardine4/', 'output/gabardine4_0.95_100_down04/data.csv', 80)
#
#volumes_felt1, X_felt1, Y_felt1 = load_data_from_npy('input/felt/felt1/', 'output/felt1_0.95_100_down04/data.csv', 15)
#volumes_felt2, X_felt2, Y_felt2 = load_data_from_npy('input/felt/felt2/', 'output/felt2_0.95_100_down04/data.csv', 30)
#volumes_felt3, X_felt3, Y_felt3 = load_data_from_npy('input/felt/felt3/', 'output/felt3_0.95_100_down04/data.csv', 45)
#volumes_felt4, X_felt4, Y_felt4 = load_data_from_npy('input/felt/felt4/', 'output/felt4_0.95_100_down04/data.csv', 60)
#volumes_felt5, X_felt5, Y_felt5 = load_data_from_npy('input/felt/felt5/', 'output/felt5_0.95_100_down04/data.csv', 75)
#volumes_felt6, X_felt6, Y_felt6 = load_data_from_npy('input/felt/felt6/', 'output/felt6_0.95_100_down04/data.csv', 90)
#volumes_felt7, X_felt7, Y_felt7 = load_data_from_npy('input/felt/felt7/', 'output/felt7_0.95_100_down04/data.csv', 105)
#volumes_felt8, X_felt8, Y_felt8 = load_data_from_npy('input/felt/felt8/', 'output/felt8_0.95_100_down04/data.csv', 120)
#volumes_felt9, X_felt9, Y_felt9 = load_data_from_npy('input/felt/felt9/', 'output/felt9_0.95_100_down04/data.csv', 135)
#volumes_felt10, X_felt10, Y_felt10 = load_data_from_npy('input/felt/felt10/', 'output/felt10_0.95_100_down04/data.csv', 150)
#volumes_felt11, X_felt11, Y_felt11 = load_data_from_npy('input/felt/felt11/', 'output/felt11_0.95_100_down04/data.csv', 165)
#

#volumes_valid = volumes_velvet9
#X_valid = X_velvet9
#Y_valid = Y_velvet9
#
##volumes_valid = volumes_gabardine4
##X_valid = X_gabardine4
##Y_valid = Y_gabardine4
#
##volumes_valid = volumes_felt5
##X_valid = X_felt5
##Y_valid = Y_felt5
#
#
#voxnet.load_weights('weights/weights_2017_5_30_12_56_38.h5')
#
#Y_pred = voxnet.predict(volumes_valid, X_valid)
#Y_pred = (1-1/(20*Y_pred))/0.95
#
#print(np.mean(abs(Y_pred-Y_valid)))
#
#filename = 'output/predictRefl.csv'
#with open(filename,'wb') as fd:    
#    np.savetxt(fd, Y_pred, delimiter=',');
#print("Save predict results to {0}".format(filename))




