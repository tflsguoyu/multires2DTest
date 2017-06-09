import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm

from multires2DRendering import multires2DRendering

def computeCorrectRefl(material,idd,blockWidth):

    filename = 'input/'+ material +'/' + material + repr(idd+1) + '.png'
    output = np.loadtxt('output/'+ material + repr(idd+1) + '_0.95_100_down04/predict.csv', delimiter=',')             
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
#    albedo = albedoMax * np.ones((idd+1,blockWidth))
    albedo = albedoMax * albedoScale.reshape(idd+1,blockWidth)
    # velvet: 15
    # gabardine: 20
    # felt: 15
    
     
    start = time.clock();
                                                        
    (downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, \
    mean_d_list, std_d_list, reflection_list, reflection_stderr_list, \
    reflectionOptimize_list, insideVis_list, albedo_k_list) \
    = multires2DRendering(filename, scale, tile, downScale, albedo, NoSamples, \
                     receiptorSize, platform, optimazation, fftOnly);
    
    print('Time elapse: ' + repr(time.clock() - start) + 's');
    
    
    N = len(downscale_list);
    
    
    ## draw curve
    # refl all
    x = np.log2(downscale_list);
    y = reflection_list[:,0];
    yerr = 1.96*reflection_stderr_list[:,0];
       
#    plt.figure();    
#    plt.errorbar(x,y,yerr=yerr,color='b',ecolor='r') 
#    plt.xlabel('log downsampleScale');
#    plt.ylabel('reflectance');
#    plt.title('Scale = ' + repr(scale) + \
#              ' Tile = ' + repr(tile) + \
#            ' Albedo = ' + repr(albedoMax) + '-' + repr(albedoMin) + \
#         ' NoSamples = ' + repr(NoSamples));
#    plt.grid(True);
#    plt.show()
    print(y)

# In[]
print('velvet start')
for idd in range(9):
    material = 'velvet'
    blockWidth = 15
    computeCorrectRefl(material,idd,blockWidth)
print('velvet end')
print(' ')
print('gabardine start')
for idd in range(4):
    material = 'gabardine'
    blockWidth = 20
    computeCorrectRefl(material,idd,blockWidth)    
print('gabardine end')
print(' ')
print('felt start')    
for idd in range(11):
    material = 'felt'
    blockWidth = 15
    computeCorrectRefl(material,idd,blockWidth)   
print('felt end')        