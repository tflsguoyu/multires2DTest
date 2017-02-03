clear; close all; clc

%%

iter = [0:1023];
for k = 900:length(iter)
    %% generate 10bits binary array
    arr10bits = dec2bin(iter(k));
    bits = length(arr10bits);
    if bits ~= 10
        for i = 1: 10-bits
            arr10bits = [num2str(0) arr10bits];
        end
    end
    bits = length(arr10bits);
    for i = 1: bits
        arr(1,i) = str2num(arr10bits(i));
    end
    sigT = repmat(arr, [bits 1]);
    filename = 'input/sigmaT_binary10bit.csv';
    csvwrite(filename, sigT);

%%  
    scale = 200;
    tile = 500;
    downScale = 1;
    NoSamples = 1000000;
    receiptorSize = 'MAX';
    optimazation = 'yes';
    numOfBlock = tile;
    platform = 'Windows_C';
    
    albedoMax = 0.65;
    albedoMin = 0.65;
    albedo = albedoMax*ones(1,numOfBlock);

%%    

    disp('');
    disp([num2str(k) '/' num2str(length(iter))]);
 
    tic;
    [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
        mean_d_list, std_d_list, reflection_list, reflection_stderr_list, ...
        reflectionOptimize_list, insideVis_list, albedo_k_list]...
    = multires2DTest(filename,scale,tile,downScale,albedo,NoSamples,...
        receiptorSize,platform,optimazation,numOfBlock);
    toc
    
%% save to file 
    output = [arr,scale,tile,NoSamples,albedoMax, ...
        reflection_list(1,1),reflection_list(2,1),albedo_k_list(2)];
    dlmwrite('results/binary10bit_0.65_200.csv',output,'delimiter',',','-append');



%%

    
%     save([filename '_results.mat']);
%     load([filename '_results.mat']);
    

%     N = length(downscale_list);

    %% frequency analysis
%     figure;
%     for i = 1: N        
%         sigmaT_d_this = sigmaT_d_list{i};
%         h = subplot(2,N,i);
%         p = get(h,'pos');
%         p(2) = p(2) - 0.2;
%         p(4) = p(4) + 0.2;
%         set(h,'pos',p);
%         imagesc(sigmaT_d_this(:,1:size(sigmaT_d_this,1)));colormap(copper);    
%         axis off
%         axis image
%         h = colorbar('southoutside');
%         t = get(h,'Limits');
%         set(h,'Ticks',linspace(t(1),t(2),2));
%         title({['mean:' num2str(mean_d_list(i))];['std:' num2str(std_d_list(i))]});
%     end    
%     for i = 1: N 
%         fftcurve_d_this = fftcurve_d_list(:,:,i);
%         h = subplot(2,N,i+N);
%         p = get(h,'pos');
%         p(2) = p(2) + 0.0;
%         p(4) = p(4) + 0.2;
%         set(h,'pos',p);
%         plot(fftcurve_d_this(1,:), fftcurve_d_this(2,:), '-');
%         xlabel('Window Size');
%         ylabel('Ratio');
%         axis equal
%         axis([0 1 0 1]);                
%     end
    
    %% refl all
%     figure(1);
%     errorbar(log2(downscale_list),reflection_list(:,1),1.96*reflection_stderr_list(:,1),'r--','LineWidth',1); hold on;
%     plot(log2(downscale_list),reflection_list(:,1),'b--','LineWidth',1);
% 
%     xlabel('log downsampleScale');
%     ylabel('reflectance');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;
    
    %% refl each
%     figure;
%     cc = jet(numOfBlock);
%     for i = 2: numOfBlock-1
%         plot(log2(downscale_list),reflection_list(:,i+1),'LineWidth',1,'Color',cc(i,:)); hold on;
%         legendInfo{i-1} = ['Block ' num2str(i)];
%     end
%     xlabel('log downsampleScale');
%     ylabel('reflectance');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;  
%     legend(legendInfo);
  
    %% optimized refl each
%     figure;
%     cc = jet(numOfBlock);
%     for i = 2: numOfBlock-1
%         plot(log2(downscale_list),reflectionOptimize_list(:,i+1),'LineWidth',1,'Color',cc(i,:)); hold on;
%         legendInfo{i-1} = ['Block ' num2str(i)];
%     end
%     xlabel('log downsampleScale');
%     ylabel('reflectance');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;  
%     legend(legendInfo);   
    
    %% optimized albedo
%     figure;
%     plot(log2(downscale_list),albedo_k_list,'*-','LineWidth',1);
%     xlabel('log downsampleScale');
%     ylabel('albedo');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo_k = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;
    
end
