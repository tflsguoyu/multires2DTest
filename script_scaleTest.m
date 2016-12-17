%% scale test
% clear;
% scale_list = [1:10];
% figure;
% for i = 1: length(scale_list)
%     [albedo_adjust,downScale,~] = func_2DTest('input/sigmaT222.csv',0.9,scale_list(i),0);
%     plot(log2(downScale),albedo_adjust,'*-'); hold on;
%     legendInfo{i} = num2str(scale_list(i));
% end
% xlabel('log downsampleScale');
% ylabel('albedo');
% legend(legendInfo);

%% albedo test
% clear;
% albedo_list = [1:-0.05:0.5];
% figure;
% for i = 1: length(albedo_list)
%     [albedo_adjust,downScale,~] = func_2DTest('input/sigmaT222.csv',albedo_list(i),1,0);
%     plot(log2(downScale),albedo_adjust,'*-'); hold on;    
% end
% xlabel('log downsampleScale');
% ylabel('albedo');


%%
clear
close all
figure;
[~,~,~] = func_2DTest('input/sigmaT_binaryRand.csv',0.95,1,1);
figure;
[~,~,~] = func_2DTest('input/sigmaT_horizontal.csv',0.95,1,1);
figure;
[~,~,~] = func_2DTest('input/sigmaT_diagonal.csv',0.95,1,1);
