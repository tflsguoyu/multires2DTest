import numpy as np

def loadSigmaT(filename):
    
    ext = filename[-3:]
    
    if ext == 'csv':
        output = np.loadtxt(filename, delimiter=',');  
    
#    if strcmp(ext,'png') || strcmp(ext,'bmp')
#        output = imread(filename);
#        if ndims(output) == 3
#            output = rgb2gray(output);
#
#        output = im2double(output); 
    
    return output
