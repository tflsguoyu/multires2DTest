% im = imread('velvet_ruffle.png');
% % im = rgb2gray(im);
% window = 32;
% step = window;
% 
% [h,w] = size(im);
% k = 0;
% count_i = 0;
% for i = 1:step:h-window+1
%     count_i = count_i + 1;
%     count_j = 0;
%     for j = 1:step:w-window+1
%         count_j = count_j + 1;
%         im_this = im(i:i+window-1, j:j+window-1);
%         imwrite(im_this, sprintf('output_ruffle/im%06d.png',k));
%         k = k + 1;
%         
%     end
%     
%     
% end
% count_i
% count_j
% 
% % im_cut = im(1:count_i*window,1:count_j*window);
% % imwrite(im_cut,'output/im.png');

% im = im2double(imread('velvet.png'));
% im1 = im(1:32*1,:);
% im2 = im(1:32*2,:);
% im3 = im(1:32*3,:);
% im4 = im(1:32*4,:);
% im5 = im(1:32*5,:);
% im6 = im(1:32*6,:);
% im7 = im(1:32*7,:);
% im8 = im(1:32*8,:);
% imwrite(im1, 'velvet1.png');
% imwrite(im2, 'velvet2.png');
% imwrite(im3, 'velvet3.png');
% imwrite(im4, 'velvet4.png');
% imwrite(im5, 'velvet5.png');
% imwrite(im6, 'velvet6.png');
% imwrite(im7, 'velvet7.png');
% imwrite(im8, 'velvet8.png');
% sum(im1(:))/size(im1,2)/32
% sum(im2(:))/size(im2,2)/32
% sum(im3(:))/size(im3,2)/32
% sum(im4(:))/size(im4,2)/32
% sum(im5(:))/size(im5,2)/32
% sum(im6(:))/size(im6,2)/32
% sum(im7(:))/size(im7,2)/32
% sum(im8(:))/size(im8,2)/32

for i = 1: 9
    im = imread(['velvet' num2str(i) '.png']);
    for j = 1: 15
        im_this = im(:,32*(j-1)+1:32*j);
        imwrite(im_this,['output' num2str(i) num2str(i) '/' sprintf('im%06d.png',j-1)]);
    end
end