from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np

def downsample(input,scale,flag):

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
        
        
input = np.tile(np.arange(1,13),(3,1))

output = downsample(input,3,'x_average');
print(input)
print(output)
print(np.mean(input));
print(np.mean(output));
