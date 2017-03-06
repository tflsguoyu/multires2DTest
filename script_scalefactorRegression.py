## Import library
# In[1]:
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.utils import np_utils
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasRegressor

## Load data
# In[2]:
def loadData(filename,numOfTrainingData):
    
    bits = 8
    output = np.loadtxt(filename+'.csv', delimiter=',');
    output[:,-5] = np.log10(output[:,-5])
    numOfTotalData = int(np.shape(output)[0])
    
    shuffle = np.arange(numOfTotalData)                   
    np.random.shuffle(shuffle)
    shuffle_train = shuffle[:numOfTrainingData]
    shuffle_test = shuffle[numOfTrainingData:]
    np.savetxt(filename+'_train'+repr(numOfTrainingData)+'_trainId.csv', shuffle_train, delimiter=',');
    np.savetxt(filename+'_train'+repr(numOfTrainingData)+'_testId.csv', shuffle_test, delimiter=',');
       
    X_train = np.c_[output[shuffle_train,:11],output[shuffle_train,13]] 
    X_all = np.c_[output[:,:11],output[:,13]] 
    
    Y_train = output[shuffle_train,16]
    Y_all = output[:,16]
        
    return (X_train, Y_train, X_all, Y_all)

## Load data
# In[2]:
def loadData3():

    filename1 = '../results/binary10bit_0.95_100.csv'
    output1 = np.loadtxt(filename1, delimiter=',');
    output1[:,10] = np.log10(output1[:,10])

    filename2 = '../results/backup/binary10bit_0.80_100.csv'
    output2 = np.loadtxt(filename2, delimiter=',');
    output2[:,10] = np.log10(output2[:,10])

    filename3 = '../results/backup/binary10bit_0.65_100.csv'
    output3 = np.loadtxt(filename3, delimiter=',');
    output3[:,10] = np.log10(output3[:,10])

    numOfTotalData = int(np.shape(output1)[0])
    numOfTrainingData = int(numOfTotalData*2)
    numOfTestingData = int(numOfTotalData)
        
    X_train1 = np.c_[output1[:,:11],output1[:,13]]
    X_train2 = np.c_[output3[:,:11],output3[:,13]]
    X_train = np.r_[X_train1,X_train2]
    Y_train = np.r_[output1[:,16],output3[:,16]]
    
    X_test = np.c_[output2[:,:11],output2[:,13]] 
    Y_test = output2[:,16]
    
        
    return (X_train, Y_train, X_test, Y_test)


## Build neural network model
# In[3]:
def buildModel(input_dim, output_dim, nb_nodes):

    model = Sequential()

    model.add(Dense(nb_nodes, input_dim=input_dim, init='normal', activation='relu')) 
#    model.add(Dense(nb_nodes, init='normal', activation='relu')) 
#    model.add(Dense(nb_nodes, init='normal', activation='relu')) 
    model.add(Dense(output_dim, init='normal'))
    
    model.compile(loss='mean_squared_error', optimizer='adam') 
    return model

## Train the model
# In[4]
def trainModel(model, X_train, Y_train, nb_epoch, X_valid,Y_valid):

    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=2,verbose=1,validation_data = (X_valid,Y_valid))
    # Plot loss trajectory throughout training.
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('method1.png')

    return model

## predict
# In[5]:
def predict(model, X):
        
    y = model.predict(X)
    y = y.reshape(y.shape[0])
    
    return y

## result
# In[6]:
def showResult(Y, y, nb_nodes, nb_epoch, filename,numOfTrainingData):
    
    np.savetxt(filename+'_train'+repr(numOfTrainingData)+'_predict.csv', y, delimiter=',');
    
    err = abs(Y-y)
          
    total = 0
    for i in range(len(Y)):
        total = total + err[i]*err[i]
    score2 = np.sqrt(total/len(Y))
    print(score2)
    
    total = 0
    for i in range(len(Y)):
        total = total + err[i]
    score1 = total/len(Y)
    print(score1)
    
    plt.figure()
    bins = np.linspace(0, 0.1, 50)
    plt.hist(err,bins)
    plt.xlabel('err')
    plt.ylabel('frequent')
    plt.title(' Data:' + repr(len(Y)) + ' Node:' + repr(nb_nodes) + ' Epoch:' 
              + repr(nb_epoch) + ' Err_L2:'+ format(score2,'.5f') + 
              '  Err_L1:'+ format(score1,'.5f'))    
#    plt.savefig(flag + '.png')
    plt.show()
    

## Main
# In[100]
filename = 'output/binary10bit_0.95_100'
numOfTrainingData = 512
(X_train, Y_train, X_all, Y_all) = loadData(filename,numOfTrainingData)

input_dim = 1
try:
    input_dim = X_train.shape[1]
except:
    pass
    
output_dim = 1
try:
    output_dim = Y_train.shape[1]
except:
    pass

print(input_dim, output_dim)

nb_nodes = 200 
nb_epoch = 100

model = buildModel(input_dim, output_dim, nb_nodes)
model = trainModel(model, X_train, Y_train, nb_epoch, X_all, Y_all)

y_all = predict(model, X_all)

showResult(Y_all,y_all, nb_nodes, nb_epoch, filename,numOfTrainingData)




