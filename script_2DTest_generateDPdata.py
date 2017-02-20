import numpy as np
import time
import matplotlib.pyplot as plt

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
    downScale = 2;
    NoSamples = 1000000;
    receiptorSize = 'MAX';
    optimazation = 'yes';
    numOfBlock = tile;
    platform = 'Windows_C';
    
    albedoMax = 0.8;
    albedoMin = 0.8;
    albedo = albedoMax * np.ones((1,numOfBlock));

    # main 
    print('');
    print(repr(iter+1) + '/' + repr(2**fullBits));
 
    start = time.clock();
    
    (downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, \
    mean_d_list, std_d_list, reflection_list, reflection_stderr_list, \
    reflectionOptimize_list, insideVis_list, albedo_k_list) \
    = multires2DTest(filename, scale, tile, downScale, albedo, NoSamples, \
                     receiptorSize, platform, optimazation, numOfBlock);
    
    print('Time elapse: ' + repr(time.clock() - start) + 's');
    
    # save to file 
    output = np.c_[ arr,scale,tile,NoSamples,albedoMax, \
        reflection_list[0,0],reflection_list[1,0],albedo_k_list[1] ];  
    with open('results/binary10bit_0.80_100.csv','ab') as fd:    
        np.savetxt(fd, output, delimiter=',');

    #