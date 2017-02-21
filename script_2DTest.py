import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

from multires2DTest import multires2DTest


filename = 'input/sigmaT_binaryRand.csv';

# %  
scale = 100;
tile = 160;
downScale = 9;
NoSamples = 10000000;
receiptorSize = 'MAX';
optimazation = 'no';
nextEvent = 'no';
numOfBlock = 20;
platform = 'Windows_C';

albedoMax = 0.95;
albedoMin = 0.95;
albedo = albedoMax * np.ones((1,numOfBlock));
 
start = time.clock();
                                                    
(downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, \
mean_d_list, std_d_list, reflection_list, reflection_stderr_list, \
reflectionOptimize_list, insideVis_list, albedo_k_list) \
= multires2DTest(filename, scale, tile, downScale, albedo, NoSamples, \
                 receiptorSize, platform, optimazation, numOfBlock);

print('Time elapse: ' + repr(time.clock() - start) + 's');


N = len(downscale_list);


## draw curve
# refl all
x = np.log2(downscale_list);
y = reflection_list[:,0];
yerr = 1.96*reflection_stderr_list[:,0];
   
plt.figure();    
plt.errorbar(x,y,yerr=yerr,fmt='r--') 
#    plt.plot(x,y,'b--');
plt.xlabel('log downsampleScale');
plt.ylabel('reflectance');
plt.title('Scale = ' + repr(scale) + \
          ' Tile = ' + repr(tile) + \
        ' Albedo = ' + repr(albedoMax) + '-' + repr(albedoMin) + \
     ' NoSamples = ' + repr(NoSamples));
plt.grid(True);
plt.show()


# refl each
start = 0.0
stop = 1.0
cc_x = np.linspace(start, stop, numOfBlock) 
cc = [ cm.jet(x) for x in cc_x ]

plt.figure();
legendInfo = [];
for i in range(1,numOfBlock+1):
    plt.plot(x,reflection_list[:,i],color=cc[i-1]); 
    legendInfo.append('Block ' + repr(i));
   
plt.xlabel('log downsampleScale');
plt.ylabel('reflectance');
plt.title('Scale = ' + repr(scale) + \
          ' Tile = ' + repr(tile) + \
        ' Albedo = ' + repr(albedoMax) + '-' + repr(albedoMin) + \
     ' NoSamples = ' + repr(NoSamples));
plt.grid(True);
plt.legend(legendInfo);  
plt.show()
