im = imread('velvet.png');
% im = rgb2gray(im);
window = 32;
step = window;

[h,w] = size(im);
k = 0;
count_i = 0;
for i = 1:step:h-window+1
    count_i = count_i + 1;
    count_j = 0;
    for j = 1:step:w-window+1
        count_j = count_j + 1;
        im_this = im(i:i+window-1, j:j+window-1);
        imwrite(im_this, sprintf('output/im%06d.png',k));
        k = k + 1;
        
    end
    
    
end
count_i
count_j

im_cut = im(1:count_i*window,1:count_j*window);
imwrite(im_cut,'output/im.png');