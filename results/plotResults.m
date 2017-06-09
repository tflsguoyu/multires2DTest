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

%% velvet

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
% deeplearn =[0.5155, 0.5061, 0.4807, 0.4732, 0.4582, 0.4434;
%            0.5286, 0.5191, 0.5001, 0.5001, 0.5030, 0.5122;
%            0.5326, 0.5238, 0.5110, 0.5142, 0.5160, 0.5228;
%            0.5354, 0.5275, 0.5207, 0.5240, 0.5266, 0.5312;
%            0.5380, 0.5307, 0.5286, 0.5316, 0.5340, 0.5364;
%            0.5402, 0.5336, 0.5355, 0.5370, 0.5391, 0.5410;
%            0.5421, 0.5358, 0.5411, 0.5413, 0.5433, 0.5442;
%            0.5436, 0.5379, 0.5454, 0.5446, 0.5471, 0.5469;
%            0.5452, 0.5396, 0.5507, 0.5493, 0.5512, 0.5496];    
% secondIdea = [0.445; 0.459; 0.468; 0.476; 0.482; 0.486; 0.49; 0.493; 0.495];
% 
%        
% figure;
% 
% plot(tile100(:,2),'g--', 'LineWidth', 0.5);hold on
% plot(tile100(:,3),'g:', 'LineWidth', 0.5);hold on
% plot(secondIdea,'y-', 'LineWidth', 0.5);hold on
% plot(tile100(:,1),'b-', 'LineWidth', 0.5);hold on
% 
% 
% 
% % plot(deeplearn(:,1),'r-', 'LineWidth', 0.5);hold on
% % plot(deeplearn(:,2),'r-', 'LineWidth', 1);hold on
% plot(deeplearn(:,3),'r-', 'LineWidth', 0.5);hold on
% plot(deeplearn(:,4),'r-', 'LineWidth', 1);hold on
% plot(deeplearn(:,5),'r-', 'LineWidth', 1.5);hold on
% plot(deeplearn(:,6),'r-', 'LineWidth', 2);hold on
%   
% grid on
% % title('Optimization')
% xlabel('Data #')
% ylabel('Reflectance')
% 
% legend('reference','16x downsample','2nd idea','3rd idea','4th idea (CNN)')

%% gabardine
%        
% deeplearn =[0.5343,0.5517,0.5004;
%             0.5540,0.5592,0.5250;
%             0.5562,0.5603,0.5285;
%             0.5575,0.5650,0.5311];    
% 
%        
% figure;
% 
% plot(deeplearn(:,1),'g--', 'LineWidth', 2);hold on
% plot(deeplearn(:,2),'g:', 'LineWidth', 2);hold on
% plot(deeplearn(:,3),'r-', 'LineWidth', 2);hold on
%   
% grid on
% % title('Optimization')
% xlabel('Data #')
% ylabel('Reflectance')
% 
% legend('reference','16x downsample','NN')

%% felt
       
% deeplearn =[0.5528,0.5584,0.5041;
%             0.5607,0.5647,0.5306;
%             0.5632,0.5664,0.5391;
%             0.5646,0.5674,0.5428;
%             0.5656,0.5680,0.5460];    
% 
%        
% figure;
% 
% plot(deeplearn(:,1),'g--', 'LineWidth', 2);hold on
% plot(deeplearn(:,2),'g:', 'LineWidth', 2);hold on
% plot(deeplearn(:,3),'r-', 'LineWidth', 2);hold on
%   
% grid on
% % title('Optimization')
% xlabel('Data #')
% ylabel('Reflectance')
% 
% legend('reference','16x downsample','NN')

% %%
% 
% velvet = [0.4546,0.5451,0.4599,0.4395;
%           0.5063,0.5581,0.5074,0.5046;
%           0.5206,0.5596,0.5144,0.5159;
%           0.5298,0.5607,0.5209,0.5256;
%           0.5361,0.5612,0.5269,0.5329;
%           0.5403,0.5618,0.5325,0.5382;
%           0.5441,0.5620,0.5373,0.5419;
%           0.5470,0.5626,0.5412,0.5452;
%           0.5490,0.5631,0.5461,0.5488];
%       
% gabardine =[0.5343,0.5517,0.5303,0.5233;
%             0.5540,0.5592,0.5464,0.5458;
%             0.5562,0.5603,0.5478,0.5479;
%             0.5575,0.5650,0.5488,0.5493];    
%        
% figure;
% subplot(1,2,1);
% plot(velvet(:,1),'g--', 'LineWidth', 2);hold on
% plot(velvet(:,2),'g:', 'LineWidth', 2);hold on
% plot(velvet(:,3),'r-', 'LineWidth', 2);hold on
% plot(velvet(:,4),'r-', 'LineWidth', 1);hold on
%   
% grid on
% title('velvet');
% xlabel('Data #')
% ylabel('Reflectance')
% legend('reference','16x downsample','NN')
% 
% subplot(1,2,2);
% plot(gabardine(:,1),'g--', 'LineWidth', 2);hold on
% plot(gabardine(:,2),'g:', 'LineWidth', 2);hold on
% plot(gabardine(:,3),'r-', 'LineWidth', 2);hold on
% plot(gabardine(:,4),'r-', 'LineWidth', 1);hold on
%   
% grid on
% title('gabardine')
% xlabel('Data #')
% ylabel('Reflectance')
% legend('reference','16x downsample','NN')

%%

velvet = [0.4546,0.5451, 0.44091003, 0.48815712, 0.53258277, 0.44528291;
          0.5063,0.5581, 0.51163347, 0.53306982, 0.54833441, 0.50393063;
          0.5206,0.5596, 0.52113432, 0.53937703, 0.55101541, 0.5159873;
          0.5298,0.5607, 0.52956763, 0.54504338, 0.55355061, 0.52507022;
          0.5361,0.5612, 0.53574824, 0.54873998, 0.55501319, 0.53229559;
          0.5403,0.5618, 0.54088011, 0.55162727, 0.55660050, 0.53785641;
          0.5441,0.5620, 0.54400449, 0.55370494, 0.55740194, 0.54163768;
          0.5470,0.5626, 0.54695754, 0.55564917, 0.55872905, 0.54502366;
          0.5490,0.5631, 0.55033267, 0.55721731, 0.55968285, 0.54902863];

gabardine =[0.5343,0.5517, 0.49454687, 0.52320548, 0.54887452, 0.52440298;
            0.5540,0.5592, 0.52946291, 0.54597619, 0.55731699, 0.54521247;
            0.5562,0.5603, 0.53311097, 0.54795605, 0.55823452, 0.54784368;
            0.5575,0.5650, 0.53642277, 0.54889633, 0.55844966, 0.54934672];   

felt =     [0.55267783,  0.55830548, 0.48671184, 0.52628856, 0.54769928, 0.54275963;
            0.56066153,  0.56494752, 0.53379644, 0.55427679, 0.55678171, 0.55678198;
            0.56295600,  0.56653195, 0.54345337, 0.55730058, 0.55955068, 0.55963440;
            0.56457247,  0.56745013, 0.54811714, 0.55936076, 0.56159797, 0.56176892;
            0.56569804,  0.56809759, 0.55170258, 0.56124710, 0.56313795, 0.56324909;
            0.56650141,  0.56835175, 0.55461629, 0.56275178, 0.56453532, 0.56443449;
            0.56694576,  0.56884707, 0.55666664, 0.56405492, 0.56541699, 0.56559071;
            0.56758210,  0.56887672, 0.55841911, 0.56477132, 0.56620258, 0.56618158;
            0.56802340,  0.56913955, 0.55968684, 0.56573889, 0.56655553, 0.56672728;
            0.56832983,  0.56938777, 0.56052221, 0.56597818, 0.56698258, 0.56705433;
            0.56843768,  0.56950120, 0.56125283, 0.56634286, 0.56723026, 0.56734120];          
figure;
plot([1:9],velvet(:,1),'g--o', 'LineWidth', 1);hold on
plot([1:9],velvet(:,2),'g:o', 'LineWidth', 1);hold on
plot([1:9],velvet(:,3),'r-+', 'LineWidth', 1);hold on
plot([1:9],velvet(:,4),'b-+', 'LineWidth', 1);hold on
plot([1:9],velvet(:,5),'m-+', 'LineWidth', 1);hold on
plot([1:9],velvet(:,6),'k-+', 'LineWidth', 1);hold on
  
plot([10:13],gabardine(:,1),'g--o', 'LineWidth', 1);hold on
plot([10:13],gabardine(:,2),'g:o', 'LineWidth', 1);hold on
plot([10:13],gabardine(:,3),'r-+', 'LineWidth', 1);hold on
plot([10:13],gabardine(:,4),'b-+', 'LineWidth', 1);hold on
plot([10:13],gabardine(:,5),'m-+', 'LineWidth', 1);hold on
plot([10:13],gabardine(:,6),'k-+', 'LineWidth', 1);hold on
% 
plot([14:24],felt(:,1),'g--o', 'LineWidth', 1);hold on
plot([14:24],felt(:,2),'g:o', 'LineWidth', 1);hold on
plot([14:24],felt(:,3),'r-+', 'LineWidth', 1);hold on
plot([14:24],felt(:,4),'b-+', 'LineWidth', 1);hold on
plot([14:24],felt(:,5),'m-+', 'LineWidth', 1);hold on
plot([14:24],felt(:,6),'k-+', 'LineWidth', 1);hold on

grid on
title('velvet | gabardine | felt');
xlabel('Data #')
ylabel('Reflectance')
legend('reference','16x downsample','NN(velvet)','NN(gabardine)','NN(felt)','NN(all three)')


% % 








