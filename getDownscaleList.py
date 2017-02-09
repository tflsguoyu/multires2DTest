import numpy as np
import math

def getDownscaleList(input, max_downscale):
    
    if max_downscale == 'MAX':
        (h, w) = np.shape(input);
        max_downscale = math.ceil(math.log2(max(h, w)));

    output = np.zeros((1,max_downscale));
    for i in range(max_downscale):
        output[0,i] = pow(2,i);
    
    return output;