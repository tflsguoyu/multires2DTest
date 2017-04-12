% a = csvread('output/velvet_0.95_100_down04_noAddLayer/velvet_0.95_100.csv');
% b = csvread('output/velvet_0.95_100_down04_addOneLayer/velvet_0.95_100.csv');
% c = csvread('output/velvet_0.95_100_down04_addTwoLayer/velvet_0.95_100.csv');
% d = csvread('output/velvet_0.95_100_down04_add3Layer/velvet_0.95_100.csv');
% e = csvread('output/velvet_0.95_100_down04_add4Layer/velvet_0.95_100.csv');
% 
% 
% plot(a(:,end),'r');hold on
% plot(b(:,end),'g');
% plot(c(:,end),'b');
% plot(d(:,end),'c');
% plot(e(:,end),'k');
% 
% 
% 
% legend('add no layer','add one layer','add two layers', 'add three layers','add four layers')

%%
% close all 
% clear
% albedo = 1-1./[5:5:25];
% scale = [1:0.5:3.5]';
% fileList = dir('output/sigmaTscaleTest/*.csv');
% 
% for j = 1: size(fileList,1)
%     a(:,:,j) = csvread(['output/sigmaTscaleTest/' fileList(j).name]);
% end
% % cmp = colormap(jet);
% % color = cmp(1:10:end,:);
% for j = 1: size(a,3)
%     fig = figure;
%     for i = 1: size(a,2)        
%         plot(scale,a(:,i,j),'*-'); hold on
%         legendInfo{i} = ['albedo: ' sprintf('%.2f',albedo(i))];
%     end
%     legend(legendInfo);
%     axis([1,4.5,0,0.6])
%     xlabel('scale (log)');
%     ylabel('reflectance');
%     imFilename = fileList(j).name;
%     imFilename = imFilename(1:end-4);
%     imFilename_fullPath = ['input/velvet/output/' imFilename '.png'];
%     im = imread(imFilename_fullPath);
%     colormap(gray)
%     imagesc([4 4.5], [0.1 0], im)
%     saveas(fig, ['output/sigmaTscaleTest/' fileList(j).name(1:end-4) '.png'],'png');
%     close all
% end

%%
close all 
clear
albedo = 1-1./[5:5:25];
scale = [1:0.5:3.5]';
fileList = dir('output/sigmaTscaleTest2/*.csv');

for j = 1: size(fileList,1)
    a(:,:,j) = csvread(['output/sigmaTscaleTest2/' fileList(j).name]);
end
% cmp = colormap(jet);
% color = cmp(1:10:end,:);
for j = 1: size(a,3)
    fig = figure;
    for i = 1: size(a,1)        
        plot(albedo,a(i,:,j),'*-'); hold on
        legendInfo{i} = ['scale: ' sprintf('%.2f',scale(i))];
    end
    legend(legendInfo);
    axis([0.8,1.05,0.8,1])
    xlabel('albedo');
    ylabel('albedo scalar factor');
    imFilename = fileList(j).name;
    imFilename = imFilename(1:end-4);
    imFilename_fullPath = ['input/velvet/output/' imFilename '.png'];
    im = imread(imFilename_fullPath);
    colormap(gray)
    imagesc([1 1.05], [0.85 0.80], im)
    saveas(fig, ['output/sigmaTscaleTest2/' fileList(j).name(1:end-4) '.png'],'png');
    close all
end