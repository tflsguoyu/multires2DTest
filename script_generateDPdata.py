import numpy as np
import time

from multires2DTest import multires2DTest

def dec2bin(x):
    return bin(x)[2:];

fullBits = 10;
for iter in range(1024):
    # generate 10bits binary array
    arr10bits = dec2bin(iter); 
    bits = len(arr10bits);
    if bits != fullBits:
        for i in range(1, fullBits-bits+1):
            arr10bits = '0' + arr10bits;         
    
    arr = np.zeros((1,fullBits));
    for i in range(fullBits):
        arr[0,i] = float(arr10bits[i]);

    sigT = np.tile(arr, (fullBits, 1));
    filename = 'input/sigmaT_binary10bit.csv';
    np.savetxt(filename, sigT, delimiter=',');

    # parameters  
    scale = 100;
    tile = 500;
    numOfBlock = tile;
    NoSamples = 1000000;
    platform = 'Windows_C';
    receiptorSize = 'MAX';
    
    downScale = 2;    
    fftOnly = 'no'
    optimazation = 'yes';
    
    albedo = 0.95
    albedoMax = albedo;
    albedoMin = albedo;
    albedo_list = albedoMax * np.ones((1,numOfBlock));

    filename_output = 'binary10bit_' + repr(albedo) + '_' + repr(scale);
    
    # main 
    print('');
    print(repr(iter+1) + '/' + repr(2**fullBits));
 
    start = time.clock();
    
    (downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, \
    mean_d_list, std_d_list, reflection_list, reflection_stderr_list, \
    reflectionOptimize_list, insideVis_list, albedo_k_list) \
    = multires2DTest(filename, scale, tile, downScale, albedo_list, NoSamples, \
                     receiptorSize, platform, optimazation, numOfBlock, fftOnly);
    
    print('Time elapse: ' + repr(time.clock() - start) + 's');

    # save to file 
    if fftOnly == 'no':
        output = np.c_[ arr,scale,tile,NoSamples,albedo, \
            reflection_list[0,0],reflection_list[1,0],albedo_k_list[1], np.mat(sigmaT_d_list[0,:10,1])/scale ];  
        with open('output/' + filename_output + '.csv','ab') as fd:    
            np.savetxt(fd, output, delimiter=',');

    #
    output2 = np.c_[ np.mat(fftcurve_d_list[0,:,0]),np.mat(fftcurve_d_list[0,:,1]) ];  
    with open('output/' + filename_output + '_freq.csv','ab') as fd:    
        np.savetxt(fd, output2, delimiter=',');