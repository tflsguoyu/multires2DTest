import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

from multires2DTest import multires2DTest



# In[]:
def computeAlbedoScalar(filename, scale, albedo):
    # %  
#    scale = 100;
    tile = 100;
    downScale = [0,4];
    NoSamples = 2000000;
    receiptorSize = 'MAX';
    fftOnly = 'no';
    optimazation = 'yes';
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
          
    print(albedo_k_list)
    return albedo_k_list[-1]                                

# In[]
for filenameID in range(2,50,5):
    filename = 'input/velvet/output/im000%03d.png' % filenameID # +15

    scaleList = np.array([10**1,10**1.5,10**2,10**2.5,10**3,10**3.5])
    albedoList = np.array([1-1/5,1-1/10,1-1/15,1-1/20,1-1/25])
    albedoScalar = np.zeros((scaleList.size, albedoList.size))
    for i,scale in enumerate(scaleList):
        for j,albedo in enumerate(albedoList):
            albedoScalar[i,j] = computeAlbedoScalar(filename, scale, albedo)
            print(scale,albedo,albedoScalar[i,j])
            
    with open('output/sigmaTscaleTest2/' + filename[-12:-4] + '.csv','ab') as fd:    
        np.savetxt(fd, albedoScalar, delimiter=',');