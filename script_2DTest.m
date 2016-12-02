function test
    clear
    close all
   
    %% Load SigmaT
    sigmaT = csvread('sigmaT3.csv');
    
    sigmaT_size = size(sigmaT,1);
    
    %% down sample
    windowsizeList = [1,2,5];
    figure;
    flag = 0;
    for windowsize = windowsizeList
        
        flag = flag + 1
        
        %% down sampling sigmaT
        window = ones(windowsize,windowsize)/(windowsize*windowsize);
        sigmaT_d_NN = conv2(sigmaT,window,'valid');
        sigmaT_d_NN = sigmaT_d_NN(1:windowsize:end,1:windowsize:end);
        csvwrite('sigmaTDownSample.csv',sigmaT_d_NN);
        
        %% std of sigmaT
        std_d(flag) = std(sigmaT_d_NN(:));
        
        subplot(3,6,flag);
        imagesc(sigmaT_d_NN,[0 6])
        colorbar
        axis equal
        axis off
        title(['std:' num2str(std_d(flag)) ' samples: ' num2str(sigmaT_size/windowsize) ' by ' num2str(sigmaT_size/windowsize)]);
                
        %% fft of sigmaT
        [fft_log_NN, fft_x(flag)] = computeFFT(sigmaT_d_NN);
        N = size(fft_log_NN,1);
        
        subplot(3,6,flag+6)
        imagesc(fft_log_NN,[0 12]);
        colorbar;hold on;
        rectangle('Position',[N*(1-fft_x(flag))/2 N*(1-fft_x(flag))/2 N*fft_x(flag) N*fft_x(flag)],'EdgeColor','w');hold off;
        axis equal
        axis off
        title(['fft:' num2str(fft_x(flag))])
        
        %% scattering    
    
        computeDensityMap('sigmaTDownSample.csv');
        densityMap = csvread('densityMap.csv');


        
        %% display densityMap
        densityMap = log(densityMap+1);
        densityMean(flag) = mean(densityMap(:));
        subplot(3,6,flag+12)
        imagesc(densityMap)
        colorbar
        axis equal
        axis off      
        title(['mean:' num2str(densityMean(flag))])
    end
    
    %% Draw curve
    figure;

    subplot(2,2,1);
    plot(windowsizeList,std_d,'*-');
    xlabel('downsampleScale');
    ylabel('std');

    subplot(2,2,2);
    plot(std_d,fft_x,'*-');
    xlabel('std');
    ylabel('fft');

    subplot(2,2,3);
    plot(std_d,densityMean,'*-');
    xlabel('std');
    ylabel('bright');

    subplot(2,2,4);
    plot(fft_x,densityMean,'*-');
    xlabel('fft');
    ylabel('bright');
    
end


function computeDensityMap(filename_sigmaT_D)

    sigmaT_d_NN = csvread(filename_sigmaT_D);

    maxDepth = 1000;
    x = zeros(maxDepth, 2);
    w = zeros(maxDepth, 2);
    y = [];
    weight = zeros(maxDepth,1);

    N_Sample = 20000;
    for samples = 1: N_Sample

        x(1,:) = [rand,1];
        w(1,:) = [0,1];
        w(2:end,:) = [];
        x(2:end,:) = [];

        albedo = 0.95;
        weight(1,1) = 1/N_Sample * albedo;
        weight(2:end,:) = [];

        for dep = 1 : maxDepth - 1
            c = round((1-x(dep,2))*size(sigmaT_d_NN,1));
            r = round(x(dep,1)*size(sigmaT_d_NN,2));
            c(c==0)=1;r(r==0)=1;
            t = -log(rand)/sigmaT_d_NN(c,r);
            x1 = x(dep, :) - t*w(dep, :);
            
            if x1(1) < 0.0 || x1(1) > 1.0 || x1(2) < 0.0 || x1(2) > 1.0
                y = [y; [x(1:dep-1,:) weight(1:dep-1,1)]];
                break;
            end
            
            theta = 2*pi*rand;
            w(dep + 1,:) = [cos(theta),sin(theta)];
            x(dep + 1,:) = x1;
            weight(dep + 1,1) = weight(dep,1)*albedo;
        end
        
    end

    % density map
    mapSize = 32;
    densityMap = zeros(mapSize,mapSize);
    for i = 1: size(y,1)
        c = round((1-y(i,2))*mapSize);
        r = round(y(i,1)*mapSize);
        c(c==0)=1;r(r==0)=1;
        densityMap(c, r) = densityMap(c, r) + y(i,3)/sigmaT_d_NN(c,r);
    end

    csvwrite('densityMap.csv',densityMap);
    
end

function [fft_log_NN,x] = computeFFT(img_NN)

    fft_NN = abs(fftshift(fft2(img_NN)));
    fft_log_NN = log(fft_NN+1);
    N = size(fft_NN,1);
    
    x_start = 0;
    x_end = 1;
    R = 0.95;
    err=1;
    while(err>0.01)
        
        x = (x_start+x_end)/2;
        M = N*x;

        fft_sub_MM = fft_NN(round(N*(1-x)/2):round(N*(1+x)/2), round(N*(1-x)/2):round(N*(1+x)/2)); 

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
end








