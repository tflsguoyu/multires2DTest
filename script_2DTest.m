clear; close all; clc

%%
filename_list{1} = 'input/sigmaT_binaryRand.csv';
% filename_list{1} = 'input/sigmaT_binaryRand_rotate.csv';

% filename_list{1} = 'input/wool.png';
% filename_list{2} = 'input/silk.png';
% filename_list{1} = 'input/sigmaT_1_J.csv';
% filename_list{1} = 'input/sigmaT_combine1.csv';
% filename_list{1} = 'input/sigmaT_combine1_albedoLeft.csv';
% filename_list{1} = 'input/sigmaT_combine1_albedoRight.csv';
% filename_list{1} = 'input/sigmaT_combine1_freqLeft.csv';
% filename_list{1} = 'input/sigmaT_combine1_freqRight.csv';
% filename_list{1} = 'input/sigmaT_combine2.csv';
% filename_list{1} = 'input/sigmaT_combine2_albedoLeft.csv';
% filename_list{1} = 'input/sigmaT_combine2_albedoRight.csv';
% filename_list{1} = 'input/sigmaT_combine2_freqLeft.csv';
% filename_list{1} = 'input/sigmaT_combine2_freqRight.csv';
%
for k = 1:length(filename_list)
    filename = filename_list{k};

    %  
    scale = 100;
    tile = 20;
    downScale = 9;
    NoSamples = 1000000;
    receiptorSize = 'MAX';
    optimazation = 'no';
    nextEvent = 'no';
    numOfBlock = 20;
    
    albedoMax = 0.95;
    albedoMin = 0.95;
    albedo = albedoMax*ones(1,numOfBlock);
    %%
%     albedoMax = 0.975;
%     albedoMin = 0.5;
%     albedo = albedoMax:(albedoMin-albedoMax)/(numOfBlock-1):albedoMin; 
    % sigmaT_combine1_albedoLeft
%     albedoMax = 0.975;
%     albedoMin = 0.75;
%     albedo = albedoMax:(albedoMin-albedoMax)/(numOfBlock-1):albedoMin; 
    % sigmaT_combine1_albedoRight
%     albedoMax = 0.725;
%     albedoMin = 0.5;
%     albedo = albedoMax:(albedoMin-albedoMax)/(numOfBlock-1):albedoMin; 
    % sigmaT_combine1_freqLeft
%     albedoMax = 0.975;
%     albedoMin = 0.5;
%     albedo = albedoMax:(albedoMin-albedoMax)/(numOfBlock*2-1):albedoMin; 
%     order = csvread('input/sigmaT_combine1_order.csv');
%     [~,I] = sort(order);
%     albedo = albedo(I);
%     albedo = albedo(1:numOfBlock);
    % sigmaT_combine1_freqRight
%     albedoMax = 0.975;
%     albedoMin = 0.5;
%     albedo = albedoMax:(albedoMin-albedoMax)/(numOfBlock*2-1):albedoMin; 
%     order = csvread('input/sigmaT_combine1_order.csv');
%     [~,I] = sort(order);
%     albedo = albedo(I);
%     albedo = albedo(numOfBlock+1:end);

%%
%     albedoMax = 0.95;
%     albedoMin = 0.65;
%     albedo = [albedoMax*ones(1,numOfBlock/2), albedoMin*ones(1,numOfBlock/2)];
%%    
%     albedoMax = 0.95;
%     albedoMin = 0.65;
%     albedo = albedoMax*ones(1,numOfBlock);
%%
%     albedoMax = 0.95;
%     albedoMin = 0.65;
%     albedo = albedoMin*ones(1,numOfBlock);
%%
%     albedoMax = 0.95;
%     albedoMin = 0.65;
%     albedo = repmat([albedoMax, albedoMin], [1 numOfBlock/2]);
%%
%     albedoMax = 0.95;
%     albedoMin = 0.65;
%     albedo = repmat([albedoMax, albedoMin], [1 numOfBlock/2]);
    

    disp('');
    disp([num2str(k) '/' num2str(length(filename_list))]);
 
    tic;
    [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
        mean_d_list, std_d_list, reflection_list, reflection_stderr_list, ...
        reflectionOptimize_list, insideVis_list, albedo_k_list]...
    = multires2DTest(filename,scale,tile,downScale,albedo,NoSamples,...
        receiptorSize,'Linux_C',optimazation,numOfBlock);
    toc
    
    % next event
    if strcmp(nextEvent,'yes')
        tic;
        [downscale_list_n, sigmaT_d_list_n, logfft_d_list_n, fftcurve_d_list_n, ...
            mean_d_list_n, std_d_list_n, reflection_list_n, reflection_stderr_list_n, ...
            reflectionOptimize_list, insideVis_list_n, albedo_k_list_n]...
        = multires2DTest(filename,scale,tile,downScale,albedo,NoSamples,...
            receiptorSize,'Windows_C_nextEvent',optimazation,numOfBlock);
        toc
    end
%     save([filename '_results.mat']);
%     load([filename '_results.mat']);
    

    N = length(downscale_list);

    
    
        %% refl all
    figure(1);
    errorbar(log2(downscale_list),reflection_list(:,1),1.96*reflection_stderr_list(:,1),'r--','LineWidth',1); hold on;
    plot(log2(downscale_list),reflection_list(:,1),'b--','LineWidth',1);

    xlabel('log downsampleScale');
    ylabel('reflectance');
    title(['Scale = ' num2str(scale) ...
        ' Tile = ' num2str(tile) ...
        ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
        ' NoSamples = ' num2str(NoSamples)]);
    grid on;
    
    %% refl each
    figure;
    cc = jet(numOfBlock);
    for i = 2: numOfBlock-1
        plot(log2(downscale_list),reflection_list(:,i+1),'LineWidth',1,'Color',cc(i,:)); hold on;
        legendInfo{i-1} = ['Block ' num2str(i)];
    end
    xlabel('log downsampleScale');
    ylabel('reflectance');
    title(['Scale = ' num2str(scale) ...
        ' Tile = ' num2str(tile) ...
        ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
        ' NoSamples = ' num2str(NoSamples)]);
    grid on;  
    legend(legendInfo);

%%
    
    
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
    
    
%     figure;
%     errorbar(log2(downscale_list),reflection_list(:,1),1.96*reflection_stderr_list(:,1),'b--','LineWidth',2); hold on;
%     plot(log2(downscale_list),reflection_list(:,1),'b--','LineWidth',2);
%     axis([0 N 0.3 0.6])
% 
%     if strcmp(nextEvent,'yes')
%         errorbar(log2(downscale_list_n),reflection_list_n(:,1),1.96*reflection_stderr_list_n(:,1),'r','LineWidth',2); hold on;
%         plot(log2(downscale_list_n),reflection_list_n(:,1),'r','LineWidth',2);
%     end
%     
%     xlabel('log downsampleScale');
%     ylabel('reflectance');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;
%     
%     order = csvread('input/sigmaT_combine1_order.csv');
%     figure;
%     cc = jet(numOfBlock);
%     for i = 1: numOfBlock
%         plot(log2(downscale_list),reflection_list(:,i+1),'LineWidth',2,'Color',cc(i,:)); hold on;
%         legendInfo{i} = ['Block ' num2str(i)];
%     end
%     xlabel('log downsampleScale');
%     ylabel('reflectance');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;  
%     legend(legendInfo);
%   
%     
%     figure;
%     %errorbar(log2(downscale_list),reflectionOptimize_list(:,1),1.96*reflection_stderr_list(:,1),'b--','LineWidth',2); hold on;
%     plot(log2(downscale_list),reflectionOptimize_list(:,1),'b--','LineWidth',2);
% 
%     if strcmp(nextEvent,'yes')
%         %errorbar(log2(downscale_list_n),reflection_list_n(:,1),1.96*reflection_stderr_list_n(:,1),'r','LineWidth',2); hold on;
%         plot(log2(downscale_list_n),reflection_list_n(:,1),'r','LineWidth',2);
%     end
%     
%     xlabel('log downsampleScale');
%     ylabel('reflectance');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;
%    
%     order = csvread('input/sigmaT_combine1_order.csv');
%     figure;
%     cc = jet(numOfBlock);
%     for i = 1: numOfBlock
%         plot(log2(downscale_list),reflectionOptimize_list(:,i+1),'LineWidth',2,'Color',cc(i,:)); hold on;
%         legendInfo{i} = ['Block ' num2str(i)];
%     end
%     xlabel('log downsampleScale');
%     ylabel('reflectance');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;  
%     legend(legendInfo);   
%     
%     figure;
%     plot(log2(downscale_list),albedo_k_list,'*-','LineWidth',2);
%     xlabel('log downsampleScale');
%     ylabel('albedo');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo_k = ' num2str(albedoMax) '-' num2str(albedoMin) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;
    
end
