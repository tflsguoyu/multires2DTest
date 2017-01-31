close all; clear; clc;
sigmaT_resolution = 256;
%% 
%     sigmaT = peaks(sigmaT_resolution)+peaks(sigmaT_resolution)';
%     sigmaT(sigmaT<0) = -sigmaT(sigmaT<0);
%     sigmaT = sigmaT * 0.5;
%     sigmaT(sigmaT<0.5) = 4-sigmaT(sigmaT<0.5);    
%%
%     while 1
%         sigmaT_row = rand(1,sigmaT_resolution);
%         sigmaT_row(sigmaT_row>0.5) = 1;
%         sigmaT_row(sigmaT_row<=0.5) = 0;   
%         a = find(sigmaT_row == 1);
%         if length(a) == 128
%             break;
%         end
%     end
%     sigmaT = repmat(sigmaT_row, [sigmaT_resolution,1]);
%%
%     sigmaT_col = rand(sigmaT_resolution,1);
%     sigmaT_col(sigmaT_col>0.5) = 1;
%     sigmaT_col(sigmaT_col<=0.5) = 0;   
%     a = find(sigmaT_col == 1);
%     length(a)
%     sigmaT = repmat(sigmaT_col, [1,sigmaT_resolution]);
%%
%     height = 128;
%     for i = 1: height
%        sigmaT_b(i,1:sigmaT_resolution) = 1; 
%     end
%     for i = 1: height
%        sigmaT_w(i,1:sigmaT_resolution) = 0; 
%     end
%     sigmaT_bw = [sigmaT_b;sigmaT_w];
%     sigmaT = repmat(sigmaT_bw,[sigmaT_resolution/(height*2),1]);
%     sigmaT = imrotate(sigmaT,90);

%%
%     sigmaT = zeros(sigmaT_resolution,sigmaT_resolution);
%     ii = 100;
%     step = 1;
%     for i = 1:step:sigmaT_resolution
%        if(ii<0)
%         ii = 100;
%        end
%         sigmaT(i:i+step-1,:) = ii;
%         ii = ii-2;       
%     end
%     sigmaT = sigmaT/100;
%     sigmaT = sigmaT(1:sigmaT_resolution, 1:sigmaT_resolution);
%     sigmaT = imrotate(sigmaT,90);

%%
% sigmaT = zeros(round(sqrt(2)*sigmaT_resolution),round(sqrt(2)*sigmaT_resolution));
% ii = 100;
% step = 1;
% for i = 1:step:size(sigmaT,1)
%    if(ii<0)
%     ii = 100;
%    end
%     sigmaT(i:i+step-1,:) = ii;
%     ii = ii-2;       
% end
% sigmaT = sigmaT/100;
% sigmaT = imrotate(sigmaT,45);
% startP = round((size(sigmaT,1)-sigmaT_resolution)/2);
% sigmaT = sigmaT(startP:startP+sigmaT_resolution-1,startP:startP+sigmaT_resolution-1);

%%
%     sigmaT = zeros(sigmaT_resolution,sigmaT_resolution);
%     ii = 1;
%     step = 10;
%     for i = 1:step:sigmaT_resolution
%        if(ii<0)
%         ii = 1;
%        end
%         sigmaT(:, i:i+step-1) = ii;
%         ii = ii-1;       
%     end
%     sigmaT = sigmaT(1:sigmaT_resolution, 1:sigmaT_resolution);
%     sigmaT = imrotate(sigmaT,90);
%%
%     sigmaT = 0.5*ones(sigmaT_resolution,sigmaT_resolution);
% %     sigmaT(1:sigmaT_resolution/2,:) = 0;
%     sigmaT = imrotate(sigmaT,90);

%%
% sigmaT = [];
% for i = 0: 19
%     row = round(sigmaT_resolution/(1.2^i));
%     while 1
%         sigmaT_row = rand(1,row);
%         sigmaT_row(sigmaT_row>0.5) = 1;
%         sigmaT_row(sigmaT_row<=0.5) = 0;   
%         a = find(sigmaT_row == 1);
%         if length(a) == round(row/2)
%             break;
%         end
%     end
%     sigmaT_row = imresize(sigmaT_row, [1 sigmaT_resolution], 'box');
%     sigmaT_this(:,:,i+1) = repmat(sigmaT_row, [sigmaT_resolution,1]);
%     figure;imshow(sigmaT_this(:,:,i+1));
% end
% order = randperm(20)
% for j = 1: 20
%    sigmaT = [sigmaT, sigmaT_this(:,:,order(j))]; 
% end
% figure;imshow(sigmaT);
% csvwrite('input/sigmaT_combine1_order.csv', order);
% dlmwrite('input/sigmaT_combine1.csv',sigmaT,'delimiter', ',', 'precision', 16);

sigmaT = [];
for i = [0, 19]
    row = round(sigmaT_resolution/(1.2^i));
    while 1
        sigmaT_row = rand(1,row);
        sigmaT_row(sigmaT_row>0.5) = 1;
        sigmaT_row(sigmaT_row<=0.5) = 0;   
        a = find(sigmaT_row == 1);
        if length(a) == round(row/2)
            break;
        end
    end
    sigmaT_row = imresize(sigmaT_row, [1 sigmaT_resolution], 'box');
    sigmaT_this(:,:,i+1) = repmat(sigmaT_row, [sigmaT_resolution,1]);
    figure;imshow(sigmaT_this(:,:,i+1));
end
sigmaT = repmat([sigmaT_this(:,:,1),sigmaT_this(:,:,20)], [1 10]);
figure;imshow(sigmaT);
hold on
plot([0:256:5120;0:256:5120],[zeros(1,21);256*ones(1,21)],'r','LineWidth',2);
dlmwrite('input/sigmaT_combine2.csv',sigmaT);

sigT_albedoLeft = sigmaT(:,1:end/2);
sigT_albedoRight = sigmaT(:,end/2+1:end);
figure;imshow(sigT_albedoLeft);
figure;imshow(sigT_albedoRight);
csvwrite('input/sigmaT_combine2_albedoLeft.csv', sigT_albedoLeft);
csvwrite('input/sigmaT_combine2_albedoRight.csv', sigT_albedoRight);

sigT_freqLeft = repmat(sigmaT_this(:,:,1), [1 10]);
sigT_freqRight = repmat(sigmaT_this(:,:,20), [1 10]);
figure;imshow(sigT_freqLeft);
figure;imshow(sigT_freqRight);
csvwrite('input/sigmaT_combine2_freqLeft.csv', sigT_freqLeft);
csvwrite('input/sigmaT_combine2_freqRight.csv', sigT_freqRight);

order = [1:20];
csvwrite('input/sigmaT_combine2_order.csv', order);
