clear; close all; clc

%%
filename_list{1} = 'input/sigmaT_binaryRand.csv';
filename_list{2} = 'input/wool.png';
filename_list{3} = 'input/silk.png';

%%
for k = 1:length(filename_list)
    filename = filename_list{k};

    %  
    scale = 1;
    tile = 0;
    albedo = 0.95;
    
    figure;
    disp('');
    disp([num2str(k) '/' num2str(length(filename_list))]);
 
% [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
%    mean_d_list, std_d_list, reflection_list, insideVis_list, albedo_list]   
    tic;
    [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
    mean_d_list, std_d_list, reflection_list, insideVis_list, albedo_list]...
    = multires2DTest(filename,scale,tile,'MAX',albedo,'Windows_C','no');
    toc
    save([filename '_results.mat']);
%     load([filename '_results.mat']);
    
    plot(log2(downscale_list),albedo_list,'*-'); hold on;
    xlabel('log downsampleScale');
    ylabel('albedo');
    title(['Scale = ' num2str(scale) ...
        ' Tile = ' num2str(tile) ...
        ' Albedo = ' num2str(albedo)]);
    grid on;
    
end
