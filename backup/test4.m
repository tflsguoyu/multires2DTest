function test
    clear
    close all
    




    %%
    sigmaT_resolution = 320;
    %%
    sigmaT = peaks(sigmaT_resolution)+peaks(sigmaT_resolution)';
    sigmaT(sigmaT<0) = -sigmaT(sigmaT<0);
    sigmaT = sigmaT * 0.5;
    sigmaT(sigmaT<0.5) = 4-sigmaT(sigmaT<0.5);    
    %%
%     sigmaT = rand(sigmaT_resolution,sigmaT_resolution)*6;

    %%
%     sigmaT = zeros(sigmaT_resolution,sigmaT_resolution);
%     ii = 6;
%     for i = 1:sigmaT_resolution
%        if(ii<0)
%         ii = 6;
%        end
%         sigmaT(i,:) = ii;
%         ii = ii-0.5;       
%     end
%     sigmaT = imrotate(sigmaT,90);

    %%
%     sigmaT = zeros(round(sqrt(2)*sigmaT_resolution),round(sqrt(2)*sigmaT_resolution));
%     ii = 6;
%     for i = 1:size(sigmaT,1)
%        if(ii<0.5)
%         ii = 6;
%        end
%         sigmaT(i,:) = ii;
%         ii = ii-0.5;       
%     end
%     sigmaT = imrotate(sigmaT,45);
%     startP = round((size(sigmaT,1)-sigmaT_resolution)/2);
%     sigmaT = sigmaT(startP:startP+sigmaT_resolution-1,startP:startP+sigmaT_resolution-1);
    
    %%    
    % down sample
    windowsizeList = [1,2,5,10,20,40];
    figure;
    flag = 0;
    for windowsize = windowsizeList
        flag = flag + 1
        window = ones(windowsize,windowsize)/(windowsize*windowsize);
        sigmaT_d = conv2(sigmaT,window,'valid');
        sigmaT_d = sigmaT_d(1:windowsize:end,1:windowsize:end);
        std_d(flag) = std(sigmaT_d(:));
        
        subplot(3,6,flag);
        imagesc(sigmaT_d,[0 6])
        colorbar
        axis equal
        axis off
        title(['std:' num2str(std_d(flag)) ' samples: ' num2str(sigmaT_resolution/windowsize) ' by ' num2str(sigmaT_resolution/windowsize)]);
        
        % fft_d = computeFFT(sigmaT_d);
        fft_d = log(abs(fftshift(fft2(sigmaT_d)))+1);
        fft_mean_d(flag) = mean(fft_d(:));

        subplot(3,6,flag+6)
        imagesc(fft_d,[0 12]);
        colorbar
        axis equal
        axis off
        title(['fft:' num2str(fft_mean_d(flag))])
        
    %%    
    
        maxDepth = 1000;
        x = zeros(maxDepth, 2);
        w = zeros(maxDepth, 2);
        y = [];
        weight = zeros(maxDepth,1);

        N_Sample = 10000;
        for samples = 1: N_Sample
            
            x(1,:) = [rand,1];
            w(1,:) = [0,1];
            w(2:end,:) = [];
            x(2:end,:) = [];
            
            albedo = 0.95;
            weight(1,1) = 1/N_Sample * albedo;
            weight(2:end,:) = [];
            
            for dep = 1 : maxDepth - 1
                c = round((1-x(dep,2))*size(sigmaT_d,1));
                r = round(x(dep,1)*size(sigmaT_d,2));
                c(c==0)=1;r(r==0)=1;
                t = -log(rand)/sigmaT_d(c,r);
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
%             y = [y; x(end,:)];
        end

        % density map
        mapSize = 32;
        densityMap = zeros(mapSize,mapSize);
        for i = 1: size(y,1)
            c = round((1-y(i,2))*mapSize);
            r = round(y(i,1)*mapSize);
            c(c==0)=1;r(r==0)=1;
            densityMap(c, r) = densityMap(c, r) + y(i,3)/sigmaT_d(c,r);
        end

%         densityMap = conv2(densityMap, ones(10,10));
%         subplot(3,6,flag+6)
%         imagesc(densityMap)
%         colorbar
%         axis equal
%         axis off
%         title(['mean:' num2str(mean(densityMap(:)))])
        
        
        densityMap = log(densityMap+1);
        densityMean(flag) = mean(densityMap(:));
        subplot(3,6,flag+12)
        imagesc(densityMap)
        colorbar
        axis equal
        axis off      
        title(['mean:' num2str(densityMean(flag))])
    end
    
    
%     std_d=[1.73,0.87223,0.34858,0.17422,0.08362,0.036857];
%     fft_mean_d=[6.0312,4.6683,2.8952,1.7031,0.77638,0.30214];
%     densityMean=[3.8622,4.2435,4.3449,4.3524,4.3495,4.3506];
%     windowsizeList = [1,2,5,10,20,40];



    figure;

    subplot(2,2,1);
    plot(windowsizeList,std_d,'*-');
    xlabel('downsampleScale');
    ylabel('std');

    subplot(2,2,2);
    plot(std_d,fft_mean_d,'*-');
    xlabel('std');
    ylabel('fft');

    subplot(2,2,3);
    plot(std_d,densityMean,'*-');
    xlabel('std');
    ylabel('bright');

    subplot(2,2,4);
    plot(fft_mean_d,densityMean,'*-');
    xlabel('fft');
    ylabel('bright');
end
