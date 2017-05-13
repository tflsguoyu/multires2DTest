import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import imread

from multires2DTest import multires2DTest



# In[]:
def computeAlbedoScalar(filename, scale, albedo):
    # %  
#    scale = 100;
    tile = 100;
    downScale = [0,4];
    NoSamples = 1000000;
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
for filenameID in range(135):
    filename = 'input/velvet/output/im000%03d.png' % filenameID # +15
    
    sigT = imread(filename, mode='F')/255
    firmBlock = np.ones((123,np.shape(sigT)[1]))
    sigT = np.r_[sigT,firmBlock]

    filename = 'input/sigmaT_tmp.csv';
    np.savetxt(filename, sigT, delimiter=',');


    albedoScalar = np.zeros((135,1))
    albedoScalar[filenameID] = computeAlbedoScalar(filename, 100, 0.95)
    print(albedoScalar[filenameID])
            
    with open('output/velvet_scaleFactor.csv','ab') as fd:    
        np.savetxt(fd, albedoScalar[filenameID], delimiter=',');