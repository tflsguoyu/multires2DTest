clear;
layer = 9;
totalImage = 15*layer;  %%%%%
data = [];
for i = 1: totalImage
   stripID = mod(i,15);
   if stripID == 0
       stripID = 15;
   end
   im_strip = [];
   for j = stripID : 15 : totalImage
       im_strip = [im_strip; im2double(imread(sprintf(['output' num2str(layer) '/im%06d.png'],j-1)))]; %%%
   end
   densityMean = sum(im_strip(:)) / 32 / 32;
   
   data = [data;[densityMean,100,0.95,0,0,0]];
   
end
csvwrite(['D:/gyDocuments/multires2DTest/output/velvet' num2str(layer) '_0.95_100_down04/data.csv'],data);%%%%%%%%%