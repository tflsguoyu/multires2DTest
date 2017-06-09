%%
% im = imread(['gabardine.png']);
% im = rgb2gray(im);
% for i = 1: 4
%    im_this = im(1:i*32,:); 
%     imwrite(im_this, ['gabardine' num2str(i) '.png']);
% 
% end
%%
% for i = 1: 4
%     im = imread(['gabardine' num2str(i) '.png']);
%     id = 0;
%     for j = 1: size(im,1)/32
%         for k = 1: size(im,2)/32
%             id = id + 1;
%             im_this = im(32*(j-1)+1:32*j, 32*(k-1)+1:32*k);
%             imwrite(im_this,['gabardine' num2str(i) '/' sprintf('im%06d.png',id-1)]);
%         end
%     end
% end
%%
im = imread(['gabardine4.png']);
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
        imwrite(im_this, sprintf('gabardine_deeplearning/im%06d.png',k));
        k = k + 1;
        
    end
    
    
end
count_i
count_j
