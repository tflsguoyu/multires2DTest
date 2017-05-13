import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

from multires2DTest import multires2DTest


filename = 'input/velvet/velvet9.png'
#filename = 'input/wool/wool.png'
#
#output = np.loadtxt('output/velvet99_0.95_100_down04/data'+'.csv', delimiter=',')
#albedoScale = output[:,-1]    
output = np.loadtxt('output/predictRefl'+'.csv', delimiter=',')             
albedoScale = output 

# %  
scale = 100;
tile = 100;
downScale = [4];
NoSamples = 10000000;
receiptorSize = 'MAX';
fftOnly = 'no';
optimazation = 'no';
platform = 'Windows_C';

albedoMax = 0.95;
albedoMin = 0.95;
#albedo = albedoMax * np.ones((1,15))
albedo = albedoMax * albedoScale.reshape(9,15)
#albedo = albedoMax * np.ones((4,21))
#albedo = albedoMax * albedoScale.reshape(4,21)

 
start = time.clock();
                                                    
(downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, \
mean_d_list, std_d_list, reflection_list, reflection_stderr_list, \
reflectionOptimize_list, insideVis_list, albedo_k_list) \
= multires2DTest(filename, scale, tile, downScale, albedo, NoSamples, \
                 receiptorSize, platform, optimazation, fftOnly);

print('Time elapse: ' + repr(time.clock() - start) + 's');


N = len(downscale_list);


## draw curve
# refl all
x = np.log2(downscale_list);
y = reflection_list[:,0];
yerr = 1.96*reflection_stderr_list[:,0];
   
plt.figure();    
plt.errorbar(x,y,yerr=yerr,color='b',ecolor='r') 
plt.xlabel('log downsampleScale');
plt.ylabel('reflectance');
plt.title('Scale = ' + repr(scale) + \
          ' Tile = ' + repr(tile) + \
        ' Albedo = ' + repr(albedoMax) + '-' + repr(albedoMin) + \
     ' NoSamples = ' + repr(NoSamples));
plt.grid(True);
plt.show()
print(y)

## refl each
#start = 0.0
#stop = 1.0
#cc_x = np.linspace(start, stop, tile) 
#cc = [ cm.jet(x) for x in cc_x ]
#
#plt.figure();
#legendInfo = [];
#for i in range(tile-2):
#    plt.plot(x,reflection_list[:,i+2],color=cc[i+1]); 
#    legendInfo.append('Block ' + repr(i+2));
#   
#plt.xlabel('log downsampleScale');
#plt.ylabel('reflectance');
#plt.title('Scale = ' + repr(scale) + \
#          ' Tile = ' + repr(tile) + \
#        ' Albedo = ' + repr(albedoMax) + '-' + repr(albedoMin) + \
#     ' NoSamples = ' + repr(NoSamples));
#plt.grid(True);
#plt.legend(legendInfo);  
#plt.show()
#
## frequency 
#plt.figure()
#legendInfo = [];
#for i in range(N):
#    fftcurve_d = fftcurve_d_list[:,:,i]
##    plt.subplot(1,N,i+1)
#    plt.plot(fftcurve_d[1,:],fftcurve_d[0,:])
#    legendInfo.append('downsample ' + repr(i));
#plt.xlabel('window size');
#plt.ylabel('energy ratio');                     
#plt.axis('equal')
#plt.grid(True);
#plt.legend(legendInfo);  
#plt.show()

