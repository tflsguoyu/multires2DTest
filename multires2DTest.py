import numpy as np

from loadSigmaT import loadSigmaT
from scaleSigmaT import scaleSigmaT
from tileSigmaT import tileSigmaT
from getDownscaleList import getDownscaleList
from computeDownsampledSigmaT import computeDownsampledSigmaT
from computeScattering import computeScattering
from deleteTmpFiles import deleteTmpFiles

def multires2DTest(sigmaT_filename, scale, tile, max_downscale, albedo, 
                   NoSamples, receiptorSize, platform, optimize, numOfBlock):

    ## 
    sigmaT = loadSigmaT(sigmaT_filename);
    (h_origin,w_origin) = np.shape(sigmaT);
    sigmaT = scaleSigmaT(sigmaT, scale);
    sigmaT = tileSigmaT(sigmaT, 'x', tile);
    (h_tile,w_tile) = np.shape(sigmaT);   
    downscale_list = getDownscaleList(sigmaT, max_downscale);
    downscaleTimes = np.size(downscale_list);
                                     
    ## output define
    reflection_list = np.zeros((downscaleTimes, numOfBlock+1));
    reflection_stderr_list = np.zeros((downscaleTimes, numOfBlock+1));
    reflectionOptimize_list = np.zeros((downscaleTimes, numOfBlock+1));
    albedo_k_list = np.zeros(downscaleTimes);

    ##
    for flag in range(downscaleTimes):
        print('downsample: ' + repr(flag) + '/' + repr(downscaleTimes-1));
        
        sigmaT_d = computeDownsampledSigmaT(sigmaT, downscale_list[flag]); 
        (h_resize,w_resize) = np.shape(sigmaT_d);
#        sigmaT_d = imresize(sigmaT_d, [h_tile,w_tile], 'box');
#        sigmaT_d_list{flag} = sigmaT_d(1:h_origin,1:w_origin);
#        
#        [logfft_d, fftcurve_d, mean_d, std_d] = computeFFT(sigmaT_d);  
#        logfft_d_list{flag} = imresize(logfft_d,[size(logfft_d,1) size(logfft_d,1)]);
#        fftcurve_d_list(:,:,flag) = fftcurve_d;
#        mean_d_list(flag) = mean_d;
#        std_d_list(flag) = std_d;
#                        
        # binary search  
        albedo_k_start = 0.5;
        albedo_k_end = 1.5;
        albedo_k_tmp = 0.0;
        err = 1;
        iter = 0;
        while abs(err) > 0.0001 and (albedo_k_end - albedo_k_start) > 0.00001:
            iter = iter + 1;
            
            albedo_k_tmp = (albedo_k_start + albedo_k_end) / 2;
            
            [reflection,reflection_stderr,insideVis] = \
                computeScattering((h_tile,w_tile),(h_resize,w_resize),\
                albedo_k_tmp*albedo,NoSamples,receiptorSize,platform,numOfBlock);
                  

            if iter == 1:
               reflection_iter1 = reflection;
               reflection_stderr_iter1 = reflection_stderr;
               #insideVis_iter1 = insideVis;

            if optimize == 'no' or flag == 0:
               break

            err = reflection[0] - reflection_list[0,0];
            if err < 0: 
                albedo_k_start = albedo_k_tmp;
            else:
                albedo_k_end = albedo_k_tmp;

        # end while        

        reflection_list[flag,:] = reflection_iter1;
        reflection_stderr_list[flag,:] = reflection_stderr_iter1;
        
        reflectionOptimize_list[flag,:] = reflection;

        insideVis_list = 0;
        albedo_k_list[flag] = albedo_k_tmp;
      
        deleteTmpFiles();


    
    # for test
#    downscale_list = 1
    sigmaT_d_list = 2
    logfft_d_list = 3
    fftcurve_d_list = 4
    mean_d_list = 5
    std_d_list = 6
#    reflection_list = np.array([[1.1,2.2,3.3],
#                             [4.4,5.5,6.6]])
#    reflection_stderr_list = 9
#    reflectionOptimize_list = 10
#    insideVis_list = 11
#    albedo_k_list = np.array([10.1,20.2,30.3])
    
    return (downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list,\
            mean_d_list, std_d_list, reflection_list, reflection_stderr_list,\
            reflectionOptimize_list, insideVis_list, albedo_k_list);
