clear;
H = 9;
W = 15;
totalImage = W*H;  %%%%%
data = [];
for i = 1: totalImage
   stripID = mod(i,W);
   if stripID == 0
       stripID = W;
   end
   im_strip = [];
   for j = stripID : W : totalImage
       im_strip = [im_strip; im2double(imread(sprintf(['output' num2str(H) '/im%06d.png'],j-1)))]; %%%
   end
   densityMean = sum(im_strip(:)) / 32 / 32;
   
   data = [data;[densityMean,100,0.95,0,0,0]];
   
end
csvwrite(['D:/gyDocuments/multires2DTest/output/velvet' num2str(H) '_0.95_100_down04/data.csv'],data);%%%%%%%%%