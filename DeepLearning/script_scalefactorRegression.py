
# coding: utf-8

# # Deep Learning Quick Start: MNIST in Keras
# 

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
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
#from sklearn.metrics import mean_squared_error

## Load data
# In[2]:
def loadData():

    filename = '../results/binary10bit_0.95_100.csv'
    output = np.loadtxt(filename, delimiter=',');
    output[:,10] = np.log10(output[:,10])
    numOfTotalData = int(np.shape(output)[0])
    numOfTrainingData = int(numOfTotalData/2)
    numOfTestingData = int(numOfTotalData/2)
    
    shuffle = np.arange(numOfTotalData)                   
    np.random.shuffle(shuffle)
    shuffle_train = shuffle[:numOfTrainingData]
    shuffle_test = shuffle[numOfTrainingData:]
    
    X_train = np.c_[output[shuffle_train,:11],output[shuffle_train,13]] 
    X_test = np.c_[output[shuffle_test,:11],output[shuffle_test,13]] 
    
    Y_train = output[shuffle_train,16]
    Y_test = output[shuffle_test,16]
        
    return (X_train, Y_train, X_test, Y_test)

## Load data
# In[2]:
def loadData2():

    filename = '../results/backup/binary10bit_0.95_100.csv'
    output = np.loadtxt(filename, delimiter=',');
    output[:,10] = np.log10(output[:,10])
    numOfTotalData = int(np.shape(output)[0])
        
    X_train = np.c_[output[:,:11],output[:,13]] 
        
    Y_train = output[:,16]
        
    return (X_train, Y_train)

## Load data
# In[2]:
def loadData3():

    filename1 = '../results/backup/binary10bit_0.95_100.csv'
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

    # Simple fully-connected neural network with 2 hidden layers.
    # Including dropout layer helps avoid overfitting.
    model = Sequential()

    model.add(Dense(nb_nodes, input_dim=input_dim, init='normal', activation='relu')) 
#    model.add(Dense(200, init='normal', activation='relu')) 
    model.add(Dense(output_dim, init='normal'))
    
#    model.add(Dense(12, input_dim=input_dim, W_regularizer=l2(0.001))) # Use input_shape=(28,28) for unflattened data.
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2)) 
#    
#    model.add(Dense(12))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2))
#    
#    model.add(Dense(output_dim))
#    model.add(Activation('softmax'))
    # Use softmax layer for multi-class problems.
    
    # ## Compile the Keras model.
    # Compiling the model **builds** each layer. Keras examines the computation graph and automatically determines the size of the weight tensors at each layer. These weights are then initialized.
    # 
    # The **loss function** is the objective function being optimized, and the *categorical crossentropy* is the appropriate loss function for the *softmax* output. For *logistic* outputs use *binomial crossentropy*, and for linear outputs use *mean_squared_error*. Some notes on the math behind this can be found here: https://www.ics.uci.edu/~pjsadows/notes.pdf.
    # 
    # The **accuracy** is just a metric that we keep track of during training. Keras also allows you to define your own metrics.
    # 
    # The **optimizer** is the algorithm used to update the weights. Typical choices include *sgd*, *rmsprop*, or *adam*. ADAM is a good choice for quick results, but standard SGD is easier to debug. In this tutorial, we use the default hyperparameters for the optimization (e.g. the initial *learning rate*), but these generally need to be tuned for each problem.    
    model.compile(loss='mean_squared_error', optimizer='adam') 
    return model

## Train the model
# In[4]
def trainModel(model, X_train, Y_train, nb_epoch):

#    seed = 7
#    np.random.seed(seed)
    

    # Weights are updated one mini-batch at a time. A running average of the training loss is computed in real time, which is useful for identifying problems (e.g. the loss might explode or get stuck right). The validation loss is evaluated at the end of each epoch (without dropout).
#    model = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5,verbose=1)    
#    model.fit(X_train,Y_train)
    
#    kfold = KFold(n_splits=5, random_state=seed)
#    results = cross_val_score(clf, X_train, Y_train, cv=kfold)
#    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=2,verbose=1)
    
    return model

## predict
# In[5]:
def predict(model, X, Y):
    
#    score1 = model.score(X_test, Y_test)
#    print(score1)
    
    y = model.predict(X)
    y = y.reshape(y.shape[0])
    
    return y

## result
# In[6]:
def showResult(Y, y, flag, nb_nodes, nb_epoch):
    
    output = np.c_[Y, y]
    np.savetxt('validation.csv', output, delimiter=',');
    
    err = abs(Y-y)
          
    total = 0
    for i in range(len(Y)):
        total = total + err[i]*err[i]
    score2 = total/len(Y)
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
    plt.title(flag + ' Data:' + repr(len(Y)) + ' Node:' + repr(nb_nodes) + ' Epoch:' 
              + repr(nb_epoch) + ' ErrL2:'+ format(score2,'.5f') + 
              '  ErrL1:'+ format(score1,'.5f'))    
    plt.savefig(flag + '.png')
    plt.show()

## Main
# In[100]

#(X_train, Y_train, X_test, Y_test) = loadData()
#
#input_dim = 1
#try:
#    input_dim = X_train.shape[1]
#except:
#    pass
#    
#output_dim = 1
#try:
#    output_dim = Y_train.shape[1]
#except:
#    pass
#
#print(input_dim, output_dim)
#
#nb_nodes = 200 
#nb_epoch = 10
#
#model = buildModel(input_dim, output_dim, nb_nodes)
#model = trainModel(model, X_train, Y_train, nb_epoch)
#
#y_train = predict(model, X_train, Y_train)
#y_test = predict(model, X_test, Y_test)
#
#showResult(Y_train,y_train,'Train', nb_nodes, nb_epoch)
#showResult(Y_test,y_test,'Test', nb_nodes, nb_epoch)



## Main
# In[100]

(X_train, Y_train) = loadData2()

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
model = trainModel(model, X_train, Y_train, nb_epoch)

y_train = predict(model, X_train, Y_train)

showResult(Y_train,y_train,'Train', nb_nodes, nb_epoch)




