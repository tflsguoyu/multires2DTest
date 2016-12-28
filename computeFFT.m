function [fft_log_NN,x_list,R_list] = computeFFT(img_NN)

    fft_NN = abs(fftshift(fft2(img_NN)));
    fft_log_NN = log(fft_NN+1);
    N = size(fft_NN,1);
    
    R_list = [0:0.01:1];
    x_list = [];
    for R = R_list
        x_start = 0;
        x_end = 1;
        err=1;
        while(err>0.0001 && (x_end-x_start)>0.0001)

            x = (x_start+x_end)/2;
            M = N*x;

            fft_sub_MM = fft_NN(round(N*(1-x)/2):round(N*(1+x)/2)+1, round(N*(1-x)/2):round(N*(1+x)/2)+1); 

            E_all = sum(fft_NN(:));
            E_sub = sum(fft_sub_MM(:));
            ratio = E_sub/E_all;

            if (ratio<R)
                x_start = x;
            else
                x_end = x;
            end

            err = abs(ratio-R);

        end
        x_list = [x_list,x];
    end
end