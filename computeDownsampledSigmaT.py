import numpy as np

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
    np.savetxt('output/sigmaTDownSample.csv', output, delimiter=',');

    return output;