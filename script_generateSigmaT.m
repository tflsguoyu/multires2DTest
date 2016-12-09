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
    step = 2;
    for i = 1:step:sigmaT_resolution
       if(ii<-0.1)
        ii = 6.0;
       end
       if ii < 0.1
           ii = 0;
       end
        sigmaT(i:i+step-1,:) = ii;
        ii = ii-0.2;       
    end
    sigmaT = sigmaT(1:sigmaT_resolution, 1:sigmaT_resolution);
    sigmaT = imrotate(sigmaT,90);

%%
% sigmaT = zeros(round(sqrt(2)*sigmaT_resolution),round(sqrt(2)*sigmaT_resolution));
% ii = 6;
% step = 4;
% for i = 1:step:size(sigmaT,1)
%    if(ii<0.2)
%     ii = 6;
%    end
%     sigmaT(i:i+step-1,:) = ii;
%     ii = ii-0.2;       
% end
% sigmaT = imrotate(sigmaT,45);
% startP = round((size(sigmaT,1)-sigmaT_resolution)/2);
% sigmaT = sigmaT(startP:startP+sigmaT_resolution-1,startP:startP+sigmaT_resolution-1);

%%
%     sigmaT = zeros(sigmaT_resolution,sigmaT_resolution);
%     ii = 1;
%     step = 10;
%     for i = 1:step:sigmaT_resolution
%        if(ii<0)
%         ii = 1;
%        end
%         sigmaT(:, i:i+step-1) = ii;
%         ii = ii-1;       
%     end
%     sigmaT = sigmaT(1:sigmaT_resolution, 1:sigmaT_resolution);
%     sigmaT = imrotate(sigmaT,90);
%%
csvwrite('input/sigmaT4.csv',sigmaT);