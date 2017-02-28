import numpy as np

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
        
        output_fft_sub = output_fft[:, int(np.ceil(W*(1-x)/2))-1:int(np.floor(W*(1+x)/2+1))]; 

        E_all = np.sum(output_fft);
        E_sub = np.sum(output_fft_sub);
        if E_all > 0:
            ratio = E_sub/E_all;
        else:
            ratio = 0
            
        R_list = np.r_[R_list,ratio];
            
    return np.c_[R_list, x_list].T;