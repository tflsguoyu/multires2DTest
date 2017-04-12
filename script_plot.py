import numpy as np
import matplotlib.pyplot as plt

fontSize = 30

filename = 'output/velvet_0.95_1000'

output = np.loadtxt(filename+'.csv', delimiter=',');

trueAlbedoScale = output[:,-1]    
originRefl = output[:,-3]
downsampleRefl = output[:,-2]
              

# In[]
increaseRefl = downsampleRefl - originRefl
refl_vs_freq = np.c_[originRefl,downsampleRefl,increaseRefl,trueAlbedoScale]


# method 1: L2 norm distance
#freqDis = [];
#for i in range(fftRatio.shape[0]):
#    dis = 0
#    for j in range(101):
#        dis = dis + (fftRatio[i,j] - fftRatio[i,j+101]) * (fftRatio[i,j] - fftRatio[i,j+101])
#    dis = np.sqrt(dis)
#    freqDis = np.r_[freqDis,dis]

# method 2:
residualMean = output[:,-6]
# method 3:
    
##    
refl_vs_freq = np.c_[refl_vs_freq, residualMean]                        
#refl_vs_freq = refl_vs_freq[np.argsort(refl_vs_freq[:,4]),:]
    
plt.figure(figsize=(30,15))
x = range(1,refl_vs_freq.shape[0]+1)

plt.plot(x[:100],refl_vs_freq[:100,2],'b*-',label='increased reflectance (ratio)')
plt.plot(x[:100],refl_vs_freq[:100,4]*1,'r*-',label='residual mean L2 norm')

plt.xticks(fontsize=fontSize) 
plt.yticks(fontsize=fontSize)
plt.xlabel('#',fontsize=fontSize)
plt.title('Albedo:0.95; Scale:100',fontsize=fontSize)    
plt.legend(fontsize=fontSize)
plt.savefig(filename+'_res_VS_refl.png')
plt.show()
                    

# In[]
fftRatio = np.loadtxt(filename+'_freq.csv', delimiter=',');
x_ = np.linspace(0,1,101)
plt.figure(figsize=(30,15))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.plot(x_,fftRatio[i+101,:101],'b-',label='original fft')
    plt.plot(x_,fftRatio[i+101,101:],'r-',label='downsampled fft')
    plt.axis('off')
plt.savefig(filename+'_fft.png')
