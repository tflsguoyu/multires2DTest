% tile100 = [0.4435,0.4546,0.5451;
%            0.5006,0.5063,0.5581;
%            0.5169,0.5206,0.5596;
%            0.5271,0.5298,0.5607;
%            0.5341,0.5361,0.5612;
%            0.5392,0.5403,0.5618;
%            0.5428,0.5441,0.5620;
%            0.5456,0.5470,0.5626;
%            0.5483,0.5490,0.5631];
%        
% tile1000 =[0.4441,0.4551,0.5457;
%            0.5012,0.5073,0.5585;
%            0.5178,0.5212,0.5603;
%            0.5281,0.5305,0.5611;
%            0.5350,0.5368,0.5621;
%            0.5397,0.5412,0.5625;
%            0.5438,0.5446,0.5631;
%            0.5465,0.5475,0.5633;
%            0.5489,0.5497,0.5637];    
% 
% 
%        
% figure;
% plot(tile100(:,1),'b-');hold on
% plot(tile100(:,2),'b:');hold on
% plot(tile100(:,3),'b--');hold on
% 
% plot(tile1000(:,1),'r-');hold on
% plot(tile1000(:,2),'r:');hold on
% plot(tile1000(:,3),'r--');hold on
%   
% grid on
% title('Local optimization (vertical stripes)')
% xlabel('Layers')
% ylabel('Reflectance')
% 
% legend('Tile100 optimized',...
%        'Tile100 reference',...
%        'Tile100 16x',...
%        'Tile1000 optimized',...
%        'Tile1000 reference',...
%        'Tile1000 16x')


tile100 = [0.4435,0.4546,0.5451;
           0.5006,0.5063,0.5581;
           0.5169,0.5206,0.5596;
           0.5271,0.5298,0.5607;
           0.5341,0.5361,0.5612;
           0.5392,0.5403,0.5618;
           0.5428,0.5441,0.5620;
           0.5456,0.5470,0.5626;
           0.5483,0.5490,0.5631];
       
deeplearn =[0.5155, 0.5061, 0.4807;
           0.5286, 0.5191, 0.5001;
           0.5326, 0.5238, 0.5110;
           0.5354, 0.5275, 0.5207;
           0.5380, 0.5307, 0.5286;
           0.5402, 0.5336, 0.5355;
           0.5421, 0.5358, 0.5411;
           0.5436, 0.5379, 0.5454;
           0.5452, 0.5396, 0.5507];    
secondIdea = [0.445; 0.459; 0.468; 0.476; 0.482; 0.486; 0.49; 0.493; 0.495];

       
figure;

plot(tile100(:,2),'g--', 'LineWidth', 2);hold on
plot(tile100(:,3),'g:', 'LineWidth', 2);hold on
plot(secondIdea,'y-', 'LineWidth', 2);hold on
plot(tile100(:,1),'b-', 'LineWidth', 2);hold on


plot(deeplearn(:,3),'r-', 'LineWidth', 2);hold on
plot(deeplearn(:,2),'r-', 'LineWidth', 1);hold on
plot(deeplearn(:,1),'r-', 'LineWidth', 0.5);hold on

  
grid on
% title('Optimization')
xlabel('Data #')
ylabel('Reflectance')

legend('reference','16x downsample','2nd idea','3rd idea','4th idea (NN)')