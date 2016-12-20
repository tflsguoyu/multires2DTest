% % clear; 
% % [albedo_adjust,downScale,~] = func_2DTest('input/sigmaT_diagonal.csv',1,1,0);
% % clear
% filename_list{1} = 'input/silk3.csv';
% % filename_list{2} = 'input/sigmaT_diagonal.csv';
% % filename_list{3} = 'input/sigmaT_horizontal_sparse.csv';
% % filename_list{4} = 'input/sigmaT_diagonal_sparse.csv';
% 
% for k = 1:1
%     k
%     filename = filename_list{k};
% 
%     % albedo test
%     clearvars -except filename k filename_list
%     albedo_list = [1:-0.1:0.5];
%     figure;
%     for j = 1: length(albedo_list)
%         j
%         [albedo_adjust,downScale,~] = func_2DTest(filename,albedo_list(j),1,0);
%         plot(log2(downScale),albedo_adjust,'*-'); hold on;    
%     end
%     xlabel('log downsampleScale');
%     ylabel('albedo');
%     grid on;
% 
%     % scale test
%     clearvars -except filename k filename_list
%     scale_list = [1:5];
%     figure;
%     for i = 1: length(scale_list)
%         i
%         [albedo_adjust,downScale,~] = func_2DTest(filename,0.9,scale_list(i),0);
%         plot(log2(downScale),albedo_adjust,'*-'); hold on;
%         legendInfo{i} = num2str(scale_list(i));
%     end
%     xlabel('log downsampleScale');
%     ylabel('albedo');
%     legend(legendInfo);
%     grid on;
% 
% 
% end

%%
% clear
% close all
% figure;
% [~,~,~] = func_2DTest('input/silk1.csv',0.95,1,1);
% figure;
% [~,~,~] = func_2DTest('input/silk2.csv',0.95,1,1);
% figure;
% [~,~,~] = func_2DTest('input/wool1.csv',0.95,1,1);
% figure;
% [~,~,~] = func_2DTest('input/wool2.csv',0.95,1,1);
% figure;
% [~,~,~] = func_2DTest('input/wool3.csv',0.95,1,1);
figure;
[~,~,~] = func_2DTest('input/silk3.csv',0.95,1,1);
