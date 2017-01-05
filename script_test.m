clear;clc

% input_a = [1 2 3 4;
%             5 6 7 8;
%             9 10 11 12;
%             13 14 15 16];
input_a = [1,2,3;
           4,5,6;
           7,8,9];

mean(input_a(:))


% output_a1 = imresize(input_a,0.5,'nearest');
% mean(output_a1(:))
% output_a11 = imresize(imresize(input_a,0.5,'nearest'),[4 4],'nearest');
% mean(output_a11(:))

output_a2 = imresize(input_a,0.5,'bilinear');
mean(output_a2(:))
output_a22 = imresize(imresize(input_a,0.5,'bilinear'),size(input_a),'bilinear');
mean(output_a22(:))

output_a3 = imresize(input_a,0.5,'bicubic');
mean(output_a3(:))
output_a33 = imresize(imresize(input_a,0.5,'bicubic'),size(input_a),'bicubic');
mean(output_a33(:))

output_a4 = imresize(input_a,0.5,'box');
mean(output_a4(:))
output_a44 = imresize(imresize(input_a,0.5,'box'),size(input_a),'box');
mean(output_a44(:))



