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


def load_data_from_npy(image_dir, value_dir, N):
    
    volumes = np.zeros((N,32,32,1))
    for iter in range(N):
        volume = imread(image_dir + 'im%06d.png' % iter, mode='F')/255
        volumes[iter,:,:,0] = volume               
               
    info = np.loadtxt(value_dir, delimiter=',');         
    X = info[:,-6:-3]
    X[:,1] = np.log10(X[:,1])
#    Y = info[:,-1]
    Y = (1 / (1 - 0.95 * info[:,-1])) / 20
    print(Y)
        
    print(volumes.shape)
    print(X.shape)
    print(Y.shape)

    return volumes, X, Y

def learning_rate_schedule(epoch):
    if epoch <= 50:
        return 0.001
    else:
        return 0.001

class VoxNet(object):
    def __init__(self):
        self.optimizer = keras.optimizers.Adam(lr=0.001)

        self.lr_schedule = LearningRateScheduler(learning_rate_schedule)

        self.histories = []

        density_input = Input(shape=(32, 32, 1), name='density_input')
#        x = Convolution2D(filters=32,
#                          kernel_size=(7, 7),
#                          strides=(2, 2),
#                          data_format='channels_last',
#                          activation='relu',
#                          kernel_regularizer=l2(0.001),
#                          bias_regularizer=l2(0.001))(density_input)
  
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
        print("Layer 5: Dense shape={0}".format(density_out.shape))

        feature_input = Input(shape=(3,), name='feature_input')
        feature_out = Dense(units=20,
                  activation='relu',
                  kernel_regularizer=l2(0.0001),
                  bias_regularizer=l2(0.0001))(feature_input)
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
                                 validation_split=0.2,
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
                                 validation_split=0.2,
                                 batch_size=32,
                                 verbose=2,
                                 callbacks=[self.lr_schedule])
        self.histories.append(history)
        return self.save_weights(), history
    
    
#voxnet = VoxNet()

# In[]
voxnet = VoxNet()

# In[] training
#
volumes_0, X_0, Y_0 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_0/data.csv', 608)
volumes_05, X_05, Y_05 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_0.5/data.csv', 608)
volumes_1, X_1, Y_1 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_1/data.csv', 608)
volumes_15, X_15, Y_15 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_1.5/data.csv', 608)
volumes_2, X_2, Y_2 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_2/data.csv', 608)
volumes_25, X_25, Y_25 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_2.5/data.csv', 608)
volumes_3, X_3, Y_3 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_3/data.csv', 608)
volumes_35, X_35, Y_35 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_3.5/data.csv', 608)
volumes_4, X_4, Y_4 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_4/data.csv', 608)

#volumes_5, X_5, Y_5 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_5/data.csv', 608)
#volumes_4, X_4, Y_4 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_4/data.csv', 608)
#volumes_6, X_6, Y_6 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_6/data.csv', 608)
#volumes_7, X_7, Y_7 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_7/data.csv', 608)
#volumes_8, X_8, Y_8 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_8/data.csv', 608)

volumes_train = np.concatenate((volumes_0,volumes_05,volumes_1,volumes_15,volumes_2,volumes_25,volumes_3,volumes_35,volumes_4))
X_train = np.concatenate((X_0,X_05,X_1,X_15,X_2,X_25,X_3,X_35,X_4))
Y_train = np.concatenate((Y_0,Y_05,Y_1,Y_15,Y_2,Y_25,Y_3,Y_35,Y_4))

#volumes_train = volumes_0
#X_train = X_0
#Y_train = Y_0

#print(volumes_train.shape)
#print(X_train.shape)
#print(Y_train.shape)

weights_filename, training_history = voxnet.fit(volumes_train, X_train, Y_train, epochs=100)

# In[] predicting

#volumes_velvet1, X_velvet1, Y_velvet1 = load_data_from_npy('input/velvet/output1/', 'output/velvet1_0.95_100_down04/data.csv', 15)
#volumes_velvet2, X_velvet2, Y_velvet2 = load_data_from_npy('input/velvet/output2/', 'output/velvet2_0.95_100_down04/data.csv', 30)
#volumes_velvet3, X_velvet3, Y_velvet3 = load_data_from_npy('input/velvet/output3/', 'output/velvet3_0.95_100_down04/data.csv', 45)
#volumes_velvet4, X_velvet4, Y_velvet4 = load_data_from_npy('input/velvet/output4/', 'output/velvet4_0.95_100_down04/data.csv', 60)
#volumes_velvet5, X_velvet5, Y_velvet5 = load_data_from_npy('input/velvet/output5/', 'output/velvet5_0.95_100_down04/data.csv', 75)
#volumes_velvet6, X_velvet6, Y_velvet6 = load_data_from_npy('input/velvet/output6/', 'output/velvet6_0.95_100_down04/data.csv', 90)
#volumes_velvet7, X_velvet7, Y_velvet7 = load_data_from_npy('input/velvet/output7/', 'output/velvet7_0.95_100_down04/data.csv', 105)
#volumes_velvet8, X_velvet8, Y_velvet8 = load_data_from_npy('input/velvet/output8/', 'output/velvet8_0.95_100_down04/data.csv', 120)
#volumes_velvet9, X_velvet9, Y_velvet9 = load_data_from_npy('input/velvet/output9/', 'output/velvet9_0.95_100_down04/data.csv', 135)
#
##volumes_0, X_0, Y_0 = load_data_from_npy('input/velvet/output_deeplearning/', 'output/velvet_0.95_100_down04_0/data.csv', 608)
#
#
#volumes_valid = volumes_velvet9
#X_valid = X_velvet9
#Y_valid = Y_velvet9
#
#voxnet.load_weights('weights/weights_2017_5_8_22_57_1.h5')
#
#Y_pred = voxnet.predict(volumes_valid, X_valid)
#
#print(np.mean(abs(Y_pred-Y_valid)))
#
#filename = 'output/predictRefl.csv'
#os.remove(filename)
#with open(filename,'ab') as fd:    
#    np.savetxt(fd, Y_pred, delimiter=',');
#print("Save predict results to {0}".format(filename))




