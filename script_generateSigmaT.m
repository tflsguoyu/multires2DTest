sigmaT_resolution = 320;
%% 
%     sigmaT = peaks(sigmaT_resolution)+peaks(sigmaT_resolution)';
%     sigmaT(sigmaT<0) = -sigmaT(sigmaT<0);
%     sigmaT = sigmaT * 0.5;
%     sigmaT(sigmaT<0.5) = 4-sigmaT(sigmaT<0.5);    
%%
%     sigmaT = rand(sigmaT_resolution,sigmaT_resolution)*6;

%%
    sigmaT = zeros(sigmaT_resolution,sigmaT_resolution);
    ii = 6;
    for i = 1:sigmaT_resolution
       if(ii<0)
        ii = 6;
       end
        sigmaT(i,:) = ii;
        ii = ii-0.5;       
    end
%     sigmaT = imrotate(sigmaT,90);

%%
% sigmaT = zeros(round(sqrt(2)*sigmaT_resolution),round(sqrt(2)*sigmaT_resolution));
% ii = 6;
% for i = 1:size(sigmaT,1)
%    if(ii<0.5)
%     ii = 6;
%    end
%     sigmaT(i,:) = ii;
%     ii = ii-0.5;       
% end
% sigmaT = imrotate(sigmaT,45);
% startP = round((size(sigmaT,1)-sigmaT_resolution)/2);
% sigmaT = sigmaT(startP:startP+sigmaT_resolution-1,startP:startP+sigmaT_resolution-1);

%%
csvwrite('sigmaT1.csv',sigmaT);