import numpy as np
import time

from multires2DTest import multires2DTest

def dec2bin(x):
    return bin(x)[2:];

file = 'train'

info = np.loadtxt('output/predict_'+file+'.csv', delimiter=',');  
fullBits = 8;
for iter in range(np.shape(info)[0]):

    sigT = np.tile(info[iter,:fullBits], (fullBits, 1));
    filename = 'input/sigmaT_binary10bit.csv';
    np.savetxt(filename, sigT, delimiter=',');

    # parameters  
    scale = pow(10,info[iter,-6]);
    tile = 100;
    numOfBlock = tile;
    NoSamples = 1000000;
    platform = 'Windows_C';
    receiptorSize = 'MAX';
    
    downScale = [1];    
    fftOnly = 'no'
    optimazation = 'no';
    
    albedo = info[iter,-5] * info[iter,-1]
    albedoMax = albedo;
    albedoMin = albedo;
    albedo_list = albedoMax * np.ones((1,numOfBlock));

    # main 
    print('');
    print(repr(iter) + '/' + repr(np.shape(info)[0]));
 
    start = time.clock();
    
    (downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, \
    mean_d_list, std_d_list, reflection_list, reflection_stderr_list, \
    reflectionOptimize_list, insideVis_list, albedo_k_list) \
    = multires2DTest(filename, scale, tile, downScale, albedo_list, NoSamples, \
                     receiptorSize, platform, optimazation, numOfBlock, fftOnly);
    
    print('Time elapse: ' + repr(time.clock() - start) + 's');
    
    # save to file 
    output = np.c_[ np.mat(info[iter,:]), reflection_list[0,0], reflection_stderr_list[0,0] ];  
    with open('output/predictRefl_'+file+'.csv','ab') as fd:    
        np.savetxt(fd, output, delimiter=',');

    #