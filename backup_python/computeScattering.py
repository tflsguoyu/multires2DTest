import numpy as np
import subprocess

def computeScattering(hw_origin,hw_resize,albedo,NoSamples,receiptorSize,platform,numOfBlock):

    sigmaT_d_filename = 'output/sigmaTDownSample.csv';

    h_sigmaT_d = hw_resize[0]; w_sigmaT_d = hw_resize[1];
    h_region = 1; 
    w_region = h_region * (hw_origin[1] / hw_origin[0]);
                          
    if receiptorSize == 'MAX':
        receiptorSize = w_region;
  
    albedo_filename = 'output/albedo.csv';
    np.savetxt(albedo_filename, albedo, delimiter=',');

    
    if platform == 'Windows_C':
    # C++ windows
        subprocess.call('scatter.exe ' + sigmaT_d_filename + ' ' + \
            repr(NoSamples) + ' ' + repr(h_sigmaT_d) + ' ' + repr(w_sigmaT_d) + ' '+ \
            repr(h_region) + ' ' + repr(w_region) + ' ' + repr(receiptorSize) + ' '+ \
            repr(numOfBlock) + ' ' + albedo_filename, shell=True);

#    if platform == 'Windows_C_nextEvent':
#    # C++ windows
#        system(['scatter_nextEvent.exe ' sigmaT_d_filename ' '...
#            repr(NoSamples) ' ' repr(h_sigmaT_d) ' ' repr(w_sigmaT_d) ' '...
#            repr(h_region) ' ' repr(w_region) ' ' repr(receiptorSize) ' '...
#            repr(numOfBlock) ' ' albedo_filename]);
        
    if platform == 'Linux_C':
    # C++ Linux
        subprocess.call('./scatter_linux ' + sigmaT_d_filename + ' ' + \
            repr(NoSamples) + ' ' + repr(h_sigmaT_d) + ' ' + repr(w_sigmaT_d) + ' '+ \
            repr(h_region) + ' ' + repr(w_region) + ' ' + repr(receiptorSize) + ' '+ \
            repr(numOfBlock) + ' ' + albedo_filename, shell=True);
    
#    if exist('output/densityMap.csv', 'file') == 2 :
#        output_insideVis = csvread('output/densityMap.csv');
#    else
    output_insideVis = 0;

    output_reflection = np.loadtxt('output/reflectance.csv', delimiter=',');
    output_reflection_stderr = np.loadtxt('output/reflectanceStderr.csv', delimiter=',');
    
    # remove boundary block
    output_reflection[0] = sum(output_reflection[2:-1]);
    output_reflection_stderr[0] = sum(output_reflection_stderr[2:-1]);
                     
    
    return (output_reflection,output_reflection_stderr,output_insideVis);
    