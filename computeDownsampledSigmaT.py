import numpy as np
from scipy.misc import imresize

def computeDownsampledSigmaT(input, scale):

#    % 4 to 1 to 4
#%     output = imresize(input,1/scale,'box');
#    
#    % 2 to 1 to 2 in x direction 
#%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))],'box'),size(input),'box');
#%     output = imresize(imresize(input,[size(input,1) round(1/scale*size(input,2))]),size(input),'box');
    
    # 4 to 1   
    (a,b) = (np.shape(input)[0], 1/scale*np.shape(input)[1]);
    output = imresize(input, (a,int(round(b))), interp='bilinear', mode='F');
    
    print(repr(a) + ' x ' + repr(int(round(b))));
    np.savetxt('output/sigmaTDownSample.csv', output, delimiter=',');

    return output;