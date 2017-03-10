import numpy as np
import matplotlib.pyplot as plt

fontSize = 30

filename = 'output/predictRefl_train.csv'

output = np.loadtxt(filename, delimiter=',')
predAlbedoScale = output[:,-3]
predRefl = output[:,-2:]
trueAlbedoScale = output[:,-4]    
originRefl = output[:,-6]
downsampleRefl = output[:,-5]
               


# In[]
refl = np.c_[originRefl,downsampleRefl,predRefl]

#refl = refl[np.argsort(refl[:,0]),:]

plt.figure(figsize=(30,15))
x = range(refl.shape[0])
plt.errorbar(x,refl[:,2],yerr=2*refl[:,3],color='m',ecolor='m',label='deeplearned')
plt.plot(x,refl[:,0],'b-',label='original')
plt.plot(x,refl[:,1],'r-',label='downsampled')
plt.xticks(fontsize=fontSize)
plt.yticks(fontsize=fontSize)
plt.xlabel('#',fontsize=fontSize)
plt.ylabel('reflectance',fontsize=fontSize)
plt.title('compare',fontsize=fontSize)    
plt.legend(fontsize=fontSize)
plt.savefig(filename[:-4]+'_compare.png')
plt.show()
