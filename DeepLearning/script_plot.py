import numpy as np
import matplotlib.pyplot as plt

filename1 = '../results/binary10bit_0.95_100.csv'
output1 = np.loadtxt(filename1, delimiter=',');
data = output1[:,14:16]

filename2 = 'binary10bit_0.95_100/optimizedReflectance.csv'
output2 = np.loadtxt(filename2, delimiter=',');
data = np.c_[data, output2]
print(data.shape)

#data = data[np.argsort(data[:,0]),:]

plt.figure(figsize=(30,15))
x = range(data.shape[0])
plt.plot(x,data[:,0],'b-',label='original')
plt.plot(x,data[:,1],'r-',label='downsampled')
plt.plot(x,data[:,2],'m-',label='deeplearned')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('#',fontsize=20)
plt.ylabel('reflectance',fontsize=20)
plt.title('Train:1024; Albedo:0.95; Scale:100',fontsize=20)    
plt.legend(fontsize=20)
plt.savefig('compare.png')
plt.show()
                    
                    