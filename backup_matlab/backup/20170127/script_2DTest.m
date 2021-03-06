clear; close all; clc

%%
% filename_list{1} = 'input/sigmaT_binaryRand.csv';
% filename_list{1} = 'input/wool.png';
% filename_list{2} = 'input/silk.png';
% filename_list{1} = 'input/sigmaT_1_J.csv';
filename_list{1} = 'input/sigmaT_combine1.csv';

%
for k = 1:length(filename_list)
    filename = filename_list{k};

    %  
    scale = 100;
    tile = 1;
    albedo = 0.95;
    NoSamples = 1000000;
    downScale = 'MAX';
    receiptorSize = 'MAX';
    nextEvent = 'no';
    numOfBlock = 20;
     
    disp('');
    disp([num2str(k) '/' num2str(length(filename_list))]);
 
    tic;
    [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
        mean_d_list, std_d_list, reflection_list, ...
        reflection_stderr_list, insideVis_list, albedo_list]...
    = multires2DTest(filename,scale,tile,downScale,albedo,NoSamples,...
        receiptorSize,'Windows_C','no');
    toc
    
    % next event
    if strcmp(nextEvent,'yes')
        tic;
        [downscale_list_n, sigmaT_d_list_n, logfft_d_list_n, fftcurve_d_list_n, ...
            mean_d_list_n, std_d_list_n, reflection_list_n, ...
            reflection_stderr_list_n, insideVis_list_n, albedo_list_n]...
        = multires2DTest(filename,scale,tile,downScale,albedo,NoSamples,...
            receiptorSize,'Windows_C_nextEvent','no');
        toc
    end
%     save([filename '_results.mat']);
%     load([filename '_results.mat']);
    

    N = length(downscale_list);

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
    
    
    figure;
    errorbar(log2(downscale_list),reflection_list(:,1),1.96*reflection_stderr_list(:,1),'b--','LineWidth',2); hold on;
    plot(log2(downscale_list),reflection_list(:,1),'b--','LineWidth',2);

    if strcmp(nextEvent,'yes')
        errorbar(log2(downscale_list_n),reflection_list_n(:,1),1.96*reflection_stderr_list_n(:,1),'r','LineWidth',2); hold on;
        plot(log2(downscale_list_n),reflection_list_n(:,1),'r','LineWidth',2);
    end
    
    xlabel('log downsampleScale');
    ylabel('reflectance');
    title(['Scale = ' num2str(scale) ...
        ' Tile = ' num2str(tile) ...
        ' Albedo = ' num2str(albedo) ...
        ' NoSamples = ' num2str(NoSamples)]);
    grid on;
    
    order = csvread('input/sigmaT_combine1_order.csv');
    figure;
    cc = jet(numOfBlock);
    for i = 1: numOfBlock
        plot(log2(downscale_list),reflection_list(:,i+1),'LineWidth',2,'Color',cc(i,:)); hold on;
        legendInfo{i} = ['Block ' num2str(i)];
    end
    xlabel('log downsampleScale');
    ylabel('reflectance');
    title(['Scale = ' num2str(scale) ...
        ' Tile = ' num2str(tile) ...
        ' Albedo = ' num2str(albedo) ...
        ' NoSamples = ' num2str(NoSamples)]);
    grid on;  
    legend(legendInfo);
%     figure;
%     plot(log2(downscale_list),albedo_list,'*-','LineWidth',2);
%     xlabel('log downsampleScale');
%     ylabel('albedo');
%     title(['Scale = ' num2str(scale) ...
%         ' Tile = ' num2str(tile) ...
%         ' Albedo = ' num2str(albedo) ...
%         ' NoSamples = ' num2str(NoSamples)]);
%     grid on;
    
end
