% clear;clc
% 

sigT = csvread('input/sigmaT_combine1.csv');
sigT_albedoLeft = sigT(:,1:end/2);
sigT_albedoRight = sigT(:,end/2+1:end);
csvwrite('input/sigmaT_combine1_albedoLeft.csv', sigT_albedoLeft);
csvwrite('input/sigmaT_combine1_albedoRight.csv', sigT_albedoRight);

sigT_order = csvread('input/sigmaT_combine1_order.csv');
sigT_freqLeft = [];
for i = 1: 10
    j = find(sigT_order==i);
    sigT_freqLeft = [sigT_freqLeft, sigT(:,(j-1)*256+1:j*256)];
end
sigT_freqRight = [];
for i = 11: 20
    j = find(sigT_order==i);
    sigT_freqRight = [sigT_freqRight, sigT(:,(j-1)*256+1:j*256)];
end
csvwrite('input/sigmaT_combine1_freqLeft.csv', sigT_freqLeft);
csvwrite('input/sigmaT_combine1_freqRight.csv', sigT_freqRight);

figure;imshow(sigT_albedoLeft);
figure;imshow(sigT_albedoRight);
figure;imshow(sigT_freqLeft);
figure;imshow(sigT_freqRight);

% sigT = sigmaT_d_list{1};
% figure;
% % imshow(sigT);
% imshow(repmat(sigT,[1,20]));
% hold on
% plot([0:256:5120;0:256:5120], [zeros(1,21);256*ones(1,21)], 'r-', 'LineWidth', 2);
% 
% % input_a = [1 2 3 4;
% %             5 6 7 8;
% %             9 10 11 12;
% %             13 14 15 16];
% input_a = [1,2,3;
%            4,5,6;
%            7,8,9];
% 
% mean(input_a(:))
% 
% 
% % output_a1 = imresize(input_a,0.5,'nearest');
% % mean(output_a1(:))
% % output_a11 = imresize(imresize(input_a,0.5,'nearest'),[4 4],'nearest');
% % mean(output_a11(:))
% 
% output_a2 = imresize(input_a,0.5,'bilinear');
% mean(output_a2(:))
% output_a22 = imresize(imresize(input_a,0.5,'bilinear'),size(input_a),'bilinear');
% mean(output_a22(:))
% 
% output_a3 = imresize(input_a,0.5,'bicubic');
% mean(output_a3(:))
% output_a33 = imresize(imresize(input_a,0.5,'bicubic'),size(input_a),'bicubic');
% mean(output_a33(:))
% 
% output_a4 = imresize(input_a,0.5,'box');
% mean(output_a4(:))
% output_a44 = imresize(imresize(input_a,0.5,'box'),size(input_a),'box');
% mean(output_a44(:))
% 
% 
% 
