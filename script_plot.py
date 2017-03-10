import numpy as np
import matplotlib.pyplot as plt

fontSize = 30

filename = 'output/binary8bit_0.8_100'

output = np.loadtxt(filename+'.csv', delimiter=',');
#fftRatio = np.loadtxt(filename+'_freq.csv', delimiter=',');

trueAlbedoScale = output[:,-1]    
originRefl = output[:,-3]
downsampleRefl = output[:,-2]
              

# In[]
increaseRefl = np.c_[downsampleRefl - originRefl]
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
refl_vs_freq = refl_vs_freq[np.argsort(refl_vs_freq[:,2]),:]
    
plt.figure(figsize=(30,15))
x = range(refl_vs_freq.shape[0])

plt.plot(x,refl_vs_freq[:,2],'b-',label='increased reflectance')
plt.plot(x,refl_vs_freq[:,4],'r-',label='residual mean L2 norm')

plt.xticks(fontsize=fontSize) 
plt.yticks(fontsize=fontSize)
plt.xlabel('#',fontsize=fontSize)
plt.title('Albedo:0.95; Scale:100',fontsize=fontSize)    
plt.legend(fontsize=fontSize)
plt.savefig(filename+'_res_VS_refl.png')
plt.show()
                    