clear; 
% [albedo_adjust,downScale,~] = func_2DTest('input/sigmaT_binaryRand.csv',10,1,0.95,3,'Windows_C');
% tic;
% [albedo_adjust,downScale,~] = func_2DTest('input/wool_edit.png',10,1000,0.95,3,'Windows_C');
% toc


%%
clear
close all
clc
tic;
% filename_list{1} = 'input/sigmaT_binaryRand.csv';
% filename_list{2} = 'input/sigmaT_horizontal.csv';
% filename_list{3} = 'input/sigmaT_diagonal.csv';

% filename_list{1} = 'input/silk.png';
filename_list{1} = 'input/wool_edit.png';

tile = 10;
for k = 1:length(filename_list)
    filename = filename_list{k};

    % frequency
    figure;
    [~,~,~] = func_2DTest(filename,tile,1000,0.95,0,'Windows_C');
      
    % albedo test
    clearvars -except filename k filename_list tile
    albedo_list = [1:-0.1:0.6];
    figure;
    for j = 1: length(albedo_list)
        disp('');
        disp([num2str(k) '/' num2str(length(filename_list)) ...
            ' part 1:' num2str(j) '/' num2str(length(albedo_list))]);
        [albedo_adjust,downScale,~] = func_2DTest(filename,tile,1000,albedo_list(j),1,'Windows_C');
        plot(log2(downScale),albedo_adjust,'*-'); hold on;    
    end
    xlabel('log downsampleScale');
    ylabel('albedo');
    grid on;

    % scale test
    clearvars -except filename k filename_list tile
    scale_list = 10.^[0:3];
    figure;
    for i = 1: length(scale_list)
        disp('');
        disp([num2str(k) '/' num2str(length(filename_list)) ...
            ' part 2:' num2str(i) '/' num2str(length(scale_list))]);
        [albedo_adjust,downScale,~] = func_2DTest(filename,tile,scale_list(i),0.95,1,'Windows_C');
        plot(log2(downScale),albedo_adjust,'*-'); hold on;
        legendInfo{i} = num2str(scale_list(i));
    end
    xlabel('log downsampleScale');
    ylabel('albedo');
    legend(legendInfo);
    grid on;


end

toc