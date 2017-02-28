function test
    clear
    close all
    
    maxDepth = 1000;
    x = zeros(maxDepth, 2);
    w = zeros(maxDepth, 2);



    %%
%      sigmaT = 3*ones(1000,1000);
    %%
    sigmaT_resolution = 320;
    sigmaT = peaks(sigmaT_resolution)+peaks(sigmaT_resolution)';
    sigmaT(sigmaT<0) = -sigmaT(sigmaT<0);
    sigmaT = sigmaT * 0.5;
    sigmaT(sigmaT<0.5) = 4-sigmaT(sigmaT<0.5);    

    %%
%     sigmaT = imrotate(peaks(1000),180);
%     sigmaT(sigmaT<0) = -sigmaT(sigmaT<0); 
%     sigmaT = sigmaT * 0.5 + 1;
%     

    %%
%     sigmaT = NaN(1000,1000);
%     a1 = 5*ones(100,1000);
%     a2 = 4.5*ones(100,1000);
%     a3 = 4*ones(100,1000);
%     a4 = 3.5*ones(100,1000);
%     a5 = 3*ones(100,1000);
%     a6 = 2.5*ones(100,1000);
%     a7 = 2*ones(100,1000);
%     a8 = 1.5*ones(100,1000);
%     a9 = 1*ones(100,1000);
%     a10 = 0.5*ones(100,1000);
% 
%     sigmaT(1:10:end,:) = a1;
%     sigmaT(2:10:end,:) = a2;
%     sigmaT(3:10:end,:) = a3;
%     sigmaT(4:10:end,:) = a4;
%     sigmaT(5:10:end,:) = a5;
%     sigmaT(6:10:end,:) = a6;
%     sigmaT(7:10:end,:) = a7;
%     sigmaT(8:10:end,:) = a8;
%     sigmaT(9:10:end,:) = a9;
%     sigmaT(10:10:end,:) = a10;
    
    %%    
    % down sample
    windowsizeList = [1,10,20,40];
    figure;
    flag = 0;
    for windowsize = windowsizeList
        flag = flag + 1
%         window = ones(windowsize,windowsize)/(windowsize*windowsize);
%         sigmaT_d = conv2(sigmaT,window,'valid');
%         sigmaT_d = sigmaT_d(1:windowsize:end,1:windowsize:end);

        sigmaT_d = imresize(imresize(sigmaT,1/windowsize),windowsize,'box');

        subplot(3,4,flag);
        imagesc(sigmaT_d)
        colorbar
        axis equal
        axis off
        title(['samples: ' num2str(sigmaT_resolution/windowsize) ' by ' num2str(sigmaT_resolution/windowsize)]);
        
    %%    
        y = [];

        for samples = 1: 10000
            
            x(1,:) = [rand,1];
            w(1,:) = [0,1];
            w(2:end,:) = [];
            x(2:end,:) = [];

            for dep = 1 : maxDepth - 1
                c = round((1-x(dep,2))*size(sigmaT_d,1));
                r = round(x(dep,1)*size(sigmaT_d,2));
                c(c==0)=1;r(r==0)=1;
                t = -log(rand)/sigmaT_d(c,r);
                x1 = x(dep, :) - t*w(dep, :);
                if x1(1) < 0.0 || x1(1) > 1.0 || x1(2) < 0.0 || x1(2) > 1.0
                    y = [y; x(1:dep-1, :)];
                    break;
                end
                theta = 2*pi*rand;
                w(dep + 1,:) = [cos(theta),sin(theta)];
                x(dep + 1,:) = x1;
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
            densityMap(c, r) = densityMap(c, r) + 1;
        end

%         densityMap = conv2(densityMap, ones(10,10));
        subplot(3,4,flag+4)
        imagesc(densityMap)
        colorbar
        axis equal
        axis off
        
        densityMap = log(densityMap+1);
        subplot(3,4,flag+8)
        imagesc(densityMap)
        colorbar
        axis equal
        axis off        
    end
end
