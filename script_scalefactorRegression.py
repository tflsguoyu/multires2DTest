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
    shuffle_valid = shuffle[numOfTrainingData:]
    np.savetxt(filename+'_train'+repr(numOfTrainingData)+'_trainId.csv', shuffle_train, delimiter=',');
    np.savetxt(filename+'_train'+repr(numOfTrainingData)+'_validId.csv', shuffle_valid, delimiter=',');
   
    
    X_train = np.c_[output[shuffle_train,:11],output[shuffle_train,13]] 
    X_all = np.c_[output[:,:11],output[:,13]] 
    
    Y_train = output[shuffle_train,16]
    Y_all = output[:,16]
        
    return (X_train, Y_train, X_valid, Y_valid, All_train, All_valid)

## Load data
# In[2]:
def loadData_multi(filenames_train, filenames_valid, bits):
    
    All_train = []
    for i in range(len(filenames_train)):
        filename_this = filenames_train[i]
        if i==0:
            All_train = np.mat(np.loadtxt(filename_this, delimiter=','))
        else:
            All_train = np.r_[All_train, np.mat(np.loadtxt(filename_this, delimiter=','))]        
    All_train[:,:4*bits] = All_train[:,:4*bits] / All_train[:,-5]
    All_train[:,-5] = np.log10(All_train[:,-5])
    
    
    All_valid = []
    for i in range(len(filenames_valid)):
        filename_this = filenames_valid[i]
        if i==0:
            All_valid = np.mat(np.loadtxt(filename_this, delimiter=','))
        else:
            All_valid = np.r_[All_valid, np.mat(np.loadtxt(filename_this, delimiter=','))] 
    All_valid[:,:4*bits] = All_valid[:,:4*bits] / All_valid[:,-5]
    All_valid[:,-5] = np.log10(All_valid[:,-5])
    
    num_train = np.shape(All_train)[0]
    num_valid = np.shape(All_valid)[0]
    
    print('num_train',num_train)    
    print('num_valid',num_valid)    
    
    X_train = np.c_[All_train[:,:1*bits],All_train[:,-5],All_train[:,-4]]
    Y_train = All_train[:,-1]
    
    X_valid = np.c_[All_valid[:,:1*bits],All_valid[:,-5],All_valid[:,-4]]
    Y_valid = All_valid[:,-1]
            
    return (X_train, Y_train, X_valid, Y_valid, All_train, All_valid)


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
def trainModel(model, X_train, Y_train, X_valid, Y_valid, nb_epoch):

    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=2,verbose=1,validation_data = (X_valid,Y_valid))
    
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()

    model.save('output/myModel.h5')
    model.save_weights('output/myModel_weight.h5')

    return model

## predict
# In[5]:
def predict(model, X):
        
    y = model.predict(X)
    y = y.reshape(y.shape[0])
    
    return y

## result
# In[6]:
def saveResult(All_train, All_valid, Y_train, Y_valid, y_train, y_valid):
    
    
    np.savetxt('output/predict_train.csv', np.c_[All_train,y_train], delimiter=',');
    np.savetxt('output/predict_valid.csv', np.c_[All_valid,y_valid], delimiter=',');

    err_train = abs(Y_train-y_train)
    
    total_train = 0
    for i in range(len(Y_train)):
        total_train = total_train + err_train[i,0]*err_train[i,0]
    score2_train = np.sqrt(total_train/len(Y_train))
    print('Train L2Err: ',score2_train)
    
    total_train = 0
    for i in range(len(Y_train)):
        total_train = total_train + err_train[i,0]
    score1_train = total_train/len(Y_train)
    print('Train L1Err: ',score1_train)

    err_valid = abs(Y_valid-y_valid)
          
    total_valid = 0
    for i in range(len(Y_valid)):
        total_valid = total_valid + err_valid[i,0]*err_valid[i,0]
    score2_valid = np.sqrt(total_valid/len(Y_valid))
    print('Test L2Err: ',score2_valid)
    
    total_valid = 0
    for i in range(len(Y_valid)):
        total_valid = total_valid + err_valid[i,0]
    score1_valid = total_valid/len(Y_valid)
    print('Test L1Err: ',score1_valid)

    fontSize = 10
    plt.figure(figsize=(10,6))
    bins = np.linspace(0, 0.5, 100)
    plt.hist(err_valid[:,0],bins,alpha = 0.7,label='validation data')
    plt.hist(err_train[:,0],bins,alpha = 0.7,label='training data')
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.xlabel('err',fontsize=fontSize)
    plt.ylabel('frequent',fontsize=fontSize)
    plt.title('Training Err L2:'+ format(score2_train,'.5f') + 
          '; Validation Err L2:'+ format(score2_valid,'.5f'),fontsize=fontSize) 
    plt.legend(fontsize=fontSize)
    plt.savefig('errhist.png')
    plt.show()

    

## Main
# In[100]

# Input single file:
#filename = 'output/binary10bit_0.95_100'
#numOfTrainingData = 512
#(X_train, Y_train, X_all, Y_all) = loadData(filename,numOfTrainingData)

# Input multiple files:
filenames_train = ['output/binary8bit_0.95_100.csv','output/binary8bit_0.65_100.csv']  
filenames_valid = ['output/binary8bit_0.8_100.csv']
bits = 8  
(X_train, Y_train, X_valid, Y_valid, All_train, All_valid) = loadData_multi(filenames_train, filenames_valid, bits)

input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]

print('input_dim',input_dim)
print('output_dim',output_dim)

nb_nodes = 1024
nb_epoch = 20

model = buildModel(input_dim, output_dim, nb_nodes)
model = trainModel(model, X_train, Y_train, X_valid, Y_valid, nb_epoch)

y_train = predict(model, X_train)
y_valid = predict(model, X_valid)


saveResult(All_train, All_valid, Y_train, Y_valid, y_train, y_valid)




