import numpy as np
import time
from scipy.misc import imread

from multires2DTest import multires2DTest

def dec2bin(x):
    return bin(x)[2:];

#fullBits = 8;
#for iter in range(pow(2,fullBits)):
#    # generate 10bits binary array
#    arrbits = dec2bin(iter); 
#    bits = len(arrbits);
#    if bits != fullBits:
#        for i in range(1, fullBits-bits+1):
#            arrbits = '0' + arrbits;         
#    
#    arr = np.zeros((1,fullBits));
#    for i in range(fullBits):
#        arr[0,i] = float(arrbits[i]);
#
#    sigT = np.tile(arr, (fullBits, 1));
#    filename = 'input/sigmaT_binary'+ repr(fullBits) +'bit.csv';
#    np.savetxt(filename, sigT, delimiter=',');



def generateData(totalSample, addLayer, densityMean):
    
    fullBits = 8
    cloth = 'velvet'

    for iter in range(totalSample):
    
#        sigT = imread('input/' + cloth + '/output' + repr(addLayer) + repr(addLayer) + '/im%06d.png' % iter, mode='F')/255
#        densityMean = np.sum(sigT)/32/15/32 
        
        sigT = imread('input/' + cloth + '/output_deeplearning' + '/im%06d.png' % iter, mode='F')/255
        firmBlock = np.ones((32*addLayer+16,np.shape(sigT)[1]))
        sigT = np.r_[sigT,firmBlock]
        
        densityMean = np.sum(sigT)/32/32
        print('');
        print('addLayer: ', addLayer)
        print('densityMean: ', densityMean)                 
        
        filename = 'input/sigmaT_tmp.csv';
        np.savetxt(filename, sigT, delimiter=',');
    
        # parameters  
        scale = 100;
        tile = 100;
        NoSamples = 1000000;
        platform = 'Windows_C';
        receiptorSize = 'MAX';
        
        downScale = [0,4];    
        fftOnly = 'no'
        optimazation = 'yes';
        
        albedo = 0.95
        albedoMax = albedo;
        albedoMin = albedo;
        albedo_list = albedoMax;
    
    #    filename_output = 'binary'+ repr(fullBits) +'bit_' + repr(albedo) + '_' + repr(scale);
        filename_output = cloth + '_' + repr(albedo) + '_' + repr(scale) + '_down04_' + repr(addLayer+0.5);
#        filename_output = cloth + repr(addLayer) + repr(addLayer) + '_' + repr(albedo) + '_' + repr(scale) + '_down04';
        
        # main 
        print('');
    #    print(repr(iter+1) + '/' + repr(2**fullBits));
        print(repr(iter+1) + '/' + repr(totalSample));
     
        start = time.clock();
        
        (downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, \
        mean_d_list, std_d_list, reflection_list, reflection_stderr_list, \
        reflectionOptimize_list, insideVis_list, albedo_k_list) \
        = multires2DTest(filename, scale, tile, downScale, albedo_list, NoSamples, \
                         receiptorSize, platform, optimazation, fftOnly);
        
        print('Time elapse: ' + repr(time.clock() - start) + 's');
    
        # save to file 
        residualMean = np.linalg.norm(sigmaT_d_list[:,:,0] - sigmaT_d_list[:,:,1]) / sigmaT_d_list[:,:,0].size
    #    residualMean = np.sum(np.abs(sigmaT_d_list[:,:,0] - sigmaT_d_list[:,:,1])) / sigmaT_d_list[:,:,0].size
                                     
        
        if fftOnly == 'no':
            output = np.c_[ np.mat(sigmaT_d_list[0,:2*fullBits,0]), np.mat(sigmaT_d_list[0,:2*fullBits,1]), \
                           residualMean, densityMean, scale, albedo, \
                           reflection_list[0,0], reflection_list[1,0], albedo_k_list[1] ];  
            with open('output/' + filename_output + '/data.csv','ab') as fd:    
                np.savetxt(fd, output, delimiter=',');
    
        #   
        output2 = np.c_[np.mat(fftcurve_d_list[0,:,0]),np.mat(fftcurve_d_list[0,:,1]) ];  
        with open('output/' + filename_output + '/data_freq.csv','ab') as fd:    
            np.savetxt(fd, output2, delimiter=',');
                      
# In[]:
for i in range(9):
    N = 608
    generateData(N,i,0)
