import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

from multires2DTest import multires2DTest



# In[]:
def computeRef(filename, scale, albedo):
    # %  
#    scale = 100;
    tile = 100;
    downScale = [0];
    NoSamples = 10000000;
    receiptorSize = 'MAX';
    fftOnly = 'no';
    optimazation = 'no';
    platform = 'Windows_C';
    
    #albedoMax = 0.95;
    #albedoMin = 0.95;
#    albedo = albedoMax
    
     
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
       
    return y                                  

# In[]
for filenameID in range(15,30):
    filename = 'input/velvet/output/im000%03d.png' % filenameID # +15

    scaleList = np.array([10**1,10**1.5,10**2,10**2.5,10**3,10**3.5])
    albedoList = np.array([1-1/5,1-1/10,1-1/15,1-1/20,1-1/25])
    ref = np.zeros((scaleList.size, albedoList.size))
    for i,scale in enumerate(scaleList):
        for j,albedo in enumerate(albedoList):
            ref[i,j] = computeRef(filename, scale, albedo)
            print(scale,albedo,ref[i,j])
            
    with open('output/' + filename[-12:-4] + '.csv','ab') as fd:    
        np.savetxt(fd, ref, delimiter=',');