import numpy as np
import matplotlib.pyplot as plt

fontSize = 30

filename = 'output/binary10bit_0.95_100'
numOfTrainingData = 512

output = np.loadtxt(filename+'.csv', delimiter=',');
fftRatio = np.loadtxt(filename+'_freq.csv', delimiter=',');
predAlbedoScale = np.loadtxt(filename+'_train'+repr(numOfTrainingData)+'_predict.csv', delimiter=',');
predRefl = np.loadtxt(filename+'_train'+repr(numOfTrainingData)+'_predictReflection.csv', delimiter=',');
trainID = np.loadtxt(filename+'_train'+repr(numOfTrainingData)+'_trainId.csv', delimiter=',');
testID = np.loadtxt(filename+'_train'+repr(numOfTrainingData)+'_testId.csv', delimiter=',');

trainID = np.int_(trainID)
testID = np.int_(testID)

trueAlbedoScale = output[:,16]    
originRefl = output[:,14]
downsampleRefl = output[:,15]
               
# In[]
Y_train = trueAlbedoScale[trainID]
y_train = predAlbedoScale[trainID]
err_train = abs(Y_train-y_train)
      
total_train = 0
for i in range(len(Y_train)):
    total_train = total_train + err_train[i]*err_train[i]
score2_train = np.sqrt(total_train/len(Y_train))
print('Train L2Err: ',score2_train)

total_train = 0
for i in range(len(Y_train)):
    total_train = total_train + err_train[i]
score1_train = total_train/len(Y_train)
print('Train L1Err: ',score1_train)

if len(testID) != 0:
    Y_test = trueAlbedoScale[testID]
    y_test = predAlbedoScale[testID]
    err_test = abs(Y_test-y_test)
          
    total_test = 0
    for i in range(len(Y_test)):
        total_test = total_test + err_test[i]*err_test[i]
    score2_test = np.sqrt(total_test/len(Y_test))
    print('Test L2Err: ',score2_test)
    
    total_test = 0
    for i in range(len(Y_test)):
        total_test = total_test + err_test[i]
    score1_test = total_test/len(Y_test)
    print('Test L1Err: ',score1_test)

plt.figure(figsize=(30,15))
bins = np.linspace(0, 0.05, 100)
if len(testID) != 0:
    plt.hist(err_test,bins,alpha = 0.7,label='validation data')
plt.hist(err_train,bins,alpha = 0.7,label='training data')
plt.xticks(fontsize=fontSize)
plt.yticks(fontsize=fontSize)
plt.xlabel('err',fontsize=fontSize)
plt.ylabel('frequent',fontsize=fontSize)
if len(testID) != 0:
    plt.title('Training Data:' + repr(len(Y_train)) + '; Training Err L2:'+ format(score2_train,'.5f') + 
          '; Validation Err L2:'+ format(score2_test,'.5f'),fontsize=fontSize) 
else:
    plt.title('Training Data:' + repr(len(Y_train)) + '; Training Err L2:'+ format(score2_train,'.5f') 
          ,fontsize=fontSize)     
plt.legend(fontsize=fontSize)
plt.savefig(filename+'_train'+repr(numOfTrainingData)+'_errhist.png')
plt.show()

# In[]
refl = np.c_[originRefl,downsampleRefl,predRefl]

refl_train = refl[trainID,:]
refl_train = refl_train[np.argsort(refl_train[:,0]),:]

plt.figure(figsize=(30,15))
x = range(refl_train.shape[0])
plt.errorbar(x,refl_train[:,2],yerr=2*refl_train[:,3],color='m',ecolor='m',label='deeplearned')
plt.plot(x,refl_train[:,0],'b-',label='original')
plt.plot(x,refl_train[:,1],'r-',label='downsampled')
plt.xticks(fontsize=fontSize)
plt.yticks(fontsize=fontSize)
plt.xlabel('#',fontsize=fontSize)
plt.ylabel('reflectance',fontsize=fontSize)
plt.title('Train:'+repr(numOfTrainingData)+'; Albedo:0.95; Scale:100',fontsize=fontSize)    
plt.legend(fontsize=fontSize)
plt.savefig(filename+'_train'+repr(numOfTrainingData)+'_compare_train.png')
plt.show()

if len(testID) != 0:    
    refl_test = refl[testID,:]
    refl_test = refl_test[np.argsort(refl_test[:,0]),:]
    
    plt.figure(figsize=(30,15))
    x = range(refl_test.shape[0])
    plt.errorbar(x,refl_test[:,2],yerr=2*refl_test[:,3],color='m',ecolor='m',label='deeplearned')
    plt.plot(x,refl_test[:,0],'b-',label='original')
    plt.plot(x,refl_test[:,1],'r-',label='downsampled')
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.xlabel('#',fontsize=fontSize)
    plt.ylabel('reflectance',fontsize=fontSize)
    plt.title('Validation:'+repr(1024-numOfTrainingData)+'; Albedo:0.95; Scale:100',fontsize=fontSize)    
    plt.legend(fontsize=fontSize)
    plt.savefig(filename+'_train'+repr(numOfTrainingData)+'_compare_test.png')
    plt.show()

# In[]
increaseRefl = np.c_[downsampleRefl - originRefl]
refl_vs_freq = np.c_[originRefl,downsampleRefl,increaseRefl,trueAlbedoScale]


# method 1: L2 norm distance
freqDis = [];
for i in range(fftRatio.shape[0]):
    dis = 0
    for j in range(101):
        dis = dis + (fftRatio[i,j] - fftRatio[i,j+101]) * (fftRatio[i,j] - fftRatio[i,j+101])
    dis = np.sqrt(dis)
    freqDis = np.r_[freqDis,dis]

# method 2:
residualMean = np.linalg.norm(output[:,:10] - output[:,-10:], axis=1)
# method 3:
    
##    
refl_vs_freq = np.c_[refl_vs_freq, residualMean]                        
refl_vs_freq = refl_vs_freq[np.argsort(refl_vs_freq[:,4]),:]
    
plt.figure(figsize=(30,15))
x = range(refl_vs_freq.shape[0])

plt.plot(x,refl_vs_freq[:,2]*5,'b-',label='increased reflectance*5')
plt.plot(x,refl_vs_freq[:,4],'r-',label='residual L2 norm')

plt.xticks(fontsize=fontSize) 
plt.yticks(fontsize=fontSize)
plt.xlabel('#',fontsize=fontSize)
plt.title('Albedo:0.95; Scale:100',fontsize=fontSize)    
plt.legend(fontsize=fontSize)
plt.savefig(filename+'_freq_VS_refl.png')
plt.show()
                    