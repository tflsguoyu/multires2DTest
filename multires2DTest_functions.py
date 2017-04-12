import numpy as np
import os
import subprocess
from scipy.misc import imread

# In[]
def loadSigmaT(filename):
    
    output = 1;
    ext = filename[-3:]
    
    if ext == 'csv':
        output = np.loadtxt(filename, delimiter=',');  
                           
    
    if ext == 'png':
        output = imread(filename);
        output = output/255; 
    
    return output


# In[]
def scaleSigmaT(input, scale):
    
    output = scale * input;
    
    return output;

# In[]
def tileSigmaT(input, flag, tile):
    
    if flag == 'x': 
        if tile:
            output = np.tile(input, (1,tile));
        else:
            output = input;    
        
    return output;

# In[]
def tileAlbedo(input, flag, tile):
    
    if flag == 'x': 
        if tile:
            output = np.tile(input, (1,tile));
        else:
            output = input;    
        
    return output;

# In[]
def getDownscaleList(downscale):
    
#    if max_downscale == 'MAX':
#        (h, w) = np.shape(input);
#        max_downscale = int(np.ceil(np.log2(max(h, w))));
#
#    output = np.zeros((max_downscale));
#    for i in range(max_downscale):
#        output[i] = pow(2,i);
    output = []
    for i in range(len(downscale)):
        output.append(pow(2,int(downscale[i])))
    return output;

# In[]
def upsample(input,scale,flag):
    
    scale = int(scale);
    output = []
    if scale > 1:
        if flag == 'x_average':
            (r,c) = np.shape(input);
            output = np.zeros((r,scale*c))
            for i in range(scale):
                output[:,i::scale] = input
    else:
        output = input
        
    return output;

# In[]
def downsample(input,scale,flag):

    scale = int(scale);
    output = [];
    if scale > 1:
        if flag == 'x_average':
            (r,c) = np.shape(input);
            input_mean = np.mean(input,axis=1);
            if (c % scale) != 0:
                appended = np.tile(input_mean, (scale-c%scale,1)).T;
                input = np.c_[input, appended];
            
            c_new = int(np.shape(input)[1]/scale);
            output = np.zeros((r,c_new));
            for j in range(c_new):
                output[:,j] = np.mean(input[:,scale*j:scale*(j+1)],axis=1);
            
        elif flag == 'x_sample':
            (r,c) = np.shape(input);
            input_mean = np.mean(input,axis=1);
            if (c % scale) != 0:
                appended = np.tile(input_mean, (scale-c%scale,1)).T;
                input = np.c_[input, appended];
            
            output = input[:,::scale];
    else:
        output = input;
        
    return output;
    
# In[]
def computeDownsampledSigmaT(input, scale, flag):

#    % 4 to 1 to 4
#%     output = imresize(input,1/scale,'box');
#    
#    % 2 to 1 to 2 in x direction 
#%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))],'box'),size(input),'box');
#%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))]),size(input),'box');
    
    # 4 to 1   
#    (a,b) = (np.shape(input)[0], 1/scale*np.shape(input)[1]);
#    output = imresize(input, (a,int(round(b))), interp='bilinear', mode='F');
    output = downsample(input, scale, flag);
    (a,b) = np.shape(output);
                      
    print(repr(a) + ' x ' + repr(int(round(b))));
    np.savetxt('tmp/sigmaTDownSample.csv', output, delimiter=',');

    return output;

# In[]
def deleteTmpFiles():
    
    filename_list = ('tmp/sigmaTDownSample.csv',\
                     'tmp/reflectance.csv',\
                     'tmp/reflectanceStderr.csv',\
                     'tmp/albedo.csv');
                    
    for filename in filename_list:
        try:
            os.remove(filename);
        except OSError:
            pass
        
# In[]
#def computeFFT(input):
#    
#    output_fft = np.abs(np.fft.fftshift(np.fft.fft2(input)));
#    W = np.shape(output_fft)[1];
#    
#    R_list = np.arange(0,1.01,0.01);
#    x_list = [];
#    for R in R_list:
#        
#        x_start = 0;
#        x_end = 1;
#        err = 1;
#        while np.abs(err) > 0.0001 and (x_end - x_start) > 0.0001:
#
#            x = (x_start+x_end)/2;
#
#            output_fft_sub = output_fft[:, int(np.ceil(W*(1-x)/2))-1:int(np.floor(W*(1+x)/2+1))]; 
#
#            E_all = np.sum(output_fft);
#            E_sub = np.sum(output_fft_sub);
#            if E_all > 0:
#                ratio = E_sub/E_all;
#            else:
#                ratio = 0
#
#            err = ratio - R;
#            if err < 0:
#                x_start = x;
#            else:
#                x_end = x;
#            
#        x_list = np.r_[x_list,x];
#            
#    return np.c_[x_list, R_list].T;


def computeFFT(input):
    
    output_fft = np.abs(np.fft.fftshift(np.fft.fft2(input)));
#    output_fft = np.log1p(output_fft)
    W = np.shape(output_fft)[1];
    
    x_list = np.arange(0,1.01,0.01);
    R_list = [];
    for x in x_list:
        
        E_all = np.sum(output_fft);
        
        if x > 0: 
            output_fft_sub = output_fft[:, int(np.ceil(W*(1-x)/2)):int(np.floor(W*(1+x)/2))]; 
            E_sub = np.sum(output_fft_sub);
        else:
            E_sub = 0;
            
        if E_all > 0:
            ratio = E_sub/E_all;
        else:
            ratio = 0
            
        R_list = np.r_[R_list,ratio];
            
    return np.c_[R_list, x_list].T;

# In[]
def computeScattering(hw_origin,hw_resize,albedo,NoSamples,receiptorSize,platform):

    sigmaT_d_filename = 'tmp/sigmaTDownSample.csv';

    h_sigmaT_d = hw_resize[0]; w_sigmaT_d = hw_resize[1];
    h_region = 1; 
    w_region = h_region * (hw_origin[1] / hw_origin[0]);
                          
    if receiptorSize == 'MAX':
        receiptorSize = w_region;
  
    albedo_filename = 'tmp/albedo.csv';
    np.savetxt(albedo_filename, albedo, delimiter=',');

    [block_row,block_col] = np.shape(albedo)
#    assert(block != numOfBlock)
    
    if platform == 'Windows_C':
    # C++ windows
        subprocess.call('scatter.exe ' + sigmaT_d_filename + ' ' + \
            repr(NoSamples) + ' ' + repr(h_sigmaT_d) + ' ' + repr(w_sigmaT_d) + ' '+ \
            repr(h_region) + ' ' + repr(w_region) + ' ' + repr(receiptorSize) + ' '+ \
            repr(block_row) + ' ' + repr(block_col) + ' ' + albedo_filename, shell=True);

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
           repr(block_row) + ' ' + repr(block_col) + ' ' + albedo_filename, shell=True);
    
#    if exist('output/densityMap.csv', 'file') == 2 :
#        output_insideVis = csvread('output/densityMap.csv');
#    else
    output_insideVis = 0;

    output_reflection = np.loadtxt('tmp/reflectance.csv', delimiter=',');
    output_reflection_stderr = np.loadtxt('tmp/reflectanceStderr.csv', delimiter=',');
    
    # remove boundary block
    output_reflection[0] = sum(output_reflection[2:-1]);
                     
    
    return (output_reflection,output_reflection_stderr,output_insideVis);
