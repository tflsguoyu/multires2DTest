%%
% im = imread(['felt.png']);
% im = rgb2gray(im);
% for i = 1: 11
%    im_this = im(1:i*32,:); 
%     imwrite(im_this, ['felt' num2str(i) '.png']);
% 
% end
%%
% for i = 1: 11
%     im = imread(['felt' num2str(i) '.png']);
%     id = 0;
%     for j = 1: size(im,1)/32
%         for k = 1: size(im,2)/32
%             id = id + 1;
%             im_this = im(32*(j-1)+1:32*j, 32*(k-1)+1:32*k);
%             imwrite(im_this,['felt' num2str(i) '/' sprintf('im%06d.png',id-1)]);
%         end
%     end
% end
%%
im = imread(['felt11.png']);
window = 32;
step = 16;

[h,w] = size(im);
k = 0;
count_i = 0;
for i = 1:step:h-window+1
    count_i = count_i + 1;
    count_j = 0;
    for j = 1:step:w-window+1
        count_j = count_j + 1;
        im_this = im(i:i+window-1, j:j+window-1);
        imwrite(im_this, sprintf('felt_deeplearning/im%06d.png',k));
        k = k + 1;
        
    end
    
    
end
count_i
count_j
