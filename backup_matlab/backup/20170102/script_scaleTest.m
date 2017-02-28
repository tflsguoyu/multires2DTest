clear; close all
% [albedo_adjust,downScale,~] = func_2DTest('input/sigmaT_binaryRand.csv',1,1,0.95,0,'Windows_C');
% tic;
% [albedo_adjust,downScale,~] = func_2DTest('input/wool_edit.png',10,1,0.95,3,'Windows_C');
% toc

%%
clear
close all
clc
tic;
filename_list{1} = 'input/sigmaT_binaryRand.csv';
filename_list{2} = 'input/wool.png';
% filename_list{3} = 'input/wool_edit.png';
filename_list{3} = 'input/silk.png';
% filename_list{5} = 'input/silk_edit.png';
% filename_list{1} = 'input/sigmaT_1_G1.csv';
% filename_list{2} = 'input/sigmaT_1_H1.csv';


tile = 50;
for k = 1:length(filename_list)
    filename = filename_list{k};

    % frequency
    figure;
    [~,~,~] = func_2DTest(filename,tile,1000,0.95,0,'Windows_C');
      
    % albedo test
%     clearvars -except filename k filename_list tile
%     albedo_list = [1:-0.1:0.8];
%     figure;
%     for j = 1: length(albedo_list)
%         disp('');
%         disp([num2str(k) '/' num2str(length(filename_list)) ...
%             ' part 1:' num2str(j) '/' num2str(length(albedo_list))]);
%         [albedo_adjust,downScale,~] = func_2DTest(filename,tile,1,albedo_list(j),1,'Windows_C');
%         plot(log2(downScale),albedo_adjust,'*-'); hold on;    
%     end
%     xlabel('log downsampleScale');
%     ylabel('albedo');
%     grid on;

    % scale test
    clearvars -except filename k filename_list tile
    scale_list = 10.^[3:3];
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

%%
% filename_list{1} = 'input/sigmaT_binaryRand.csv';
% filename_list{1} = 'input/wool.png';
% filename_list{2} = 'input/wool_edit.png';
% filename_list{1} = 'input/silk.png';
% filename_list{4} = 'input/silk_edit.png';
% filename_list{6} = 'input/sigmaT_horizontal.csv';
% filename_list{7} = 'input/sigmaT_diagonal.csv';
% % filename_list{1} = 'input/sigmaT_0.5.csv';
% filename_list{1} = 'input/sigmaT_1_A.csv';
% filename_list{2} = 'input/sigmaT_1_B.csv';
% filename_list{3} = 'input/sigmaT_1_C.csv';
% filename_list{1} = 'input/sigmaT_1_D.csv';
% filename_list{5} = 'input/sigmaT_1_E.csv';
% filename_list{6} = 'input/sigmaT_1_F.csv';
% filename_list{1} = 'input/sigmaT_1_H.csv';
% filename_list{2} = 'input/sigmaT_1_H1.csv';
% filename_list{3} = 'input/sigmaT_1_H2.csv';
% filename_list{4} = 'input/sigmaT_1_H4.csv';
% filename_list{5} = 'input/sigmaT_1_H8.csv';
% filename_list{6} = 'input/sigmaT_1_H16.csv';
% filename_list{7} = 'input/sigmaT_1_H32.csv';
% filename_list{8} = 'input/sigmaT_1_H64.csv';
% filename_list{1} = 'input/sigmaT_1_J.csv';



% for k = 1:length(filename_list)
%     tic
%     filename = filename_list{k};
% 
%     %
% %     figure(1);
% %     sigmaT = csvread(filename);
% %     subplot(2,4,k);
% %     imshow(sigmaT);
% %     axis off
% %     axis image
%     
%     % tile test
%     clearvars -except filename k filename_list tic
%     tile_list = 2.^[0:7];
%     figure;
%     for j = 1: length(tile_list)
%         disp('');
%         disp([num2str(k) '/' num2str(length(filename_list)) ...
%             ' ' num2str(j) '/' num2str(length(tile_list))]);
%         [reflection(j,:),downScale,~] = func_2DTest_reflection(filename,tile_list(j),1,0.95,1,'Windows_C');
% %         plot(log2(downScale),reflection,'*-'); hold on;    
%     end
%     for i = 1: length(downScale)
%         plot(tile_list, reflection(:,i)','*-');hold on;
%         legendInfo{i} = ['downsample scale = ' num2str(downScale(i))];
%     end
%     xlabel('tile (\times N)');
%     ylabel('appearance');
%     legend(legendInfo);
%     grid on;
%     toc
% end
