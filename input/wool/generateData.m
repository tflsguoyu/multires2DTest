im = imread('wool.png');
% im = rgb2gray(im);
% imwrite(im, 'wool.png')
window = 32;
step = 32;

[h,w] = size(im);
k = 0;
for i = 1:step:h-window+1
    i
    for j = 1:step:w-window+1
        j
        im_this = im(i:i+window-1, j:j+window-1);
        imwrite(im_this, sprintf('output/im%06d.png',k));
        k = k + 1;
        
    end
    
    
end