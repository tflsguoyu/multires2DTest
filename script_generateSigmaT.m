sigmaT_resolution = 256;
%% 
%     sigmaT = peaks(sigmaT_resolution)+peaks(sigmaT_resolution)';
%     sigmaT(sigmaT<0) = -sigmaT(sigmaT<0);
%     sigmaT = sigmaT * 0.5;
%     sigmaT(sigmaT<0.5) = 4-sigmaT(sigmaT<0.5);    
%%
%     sigmaT_row = rand(1,sigmaT_resolution);
%     sigmaT_row(sigmaT_row>0.5) = 1;
%     sigmaT_row(sigmaT_row<=0.5) = 0;   
%     sigmaT = repmat(sigmaT_row, [sigmaT_resolution,1]);

%%
%     sigmaT = zeros(sigmaT_resolution,sigmaT_resolution);
%     ii = 100;
%     step = 1;
%     for i = 1:step:sigmaT_resolution
%        if(ii<0)
%         ii = 100;
%        end
%         sigmaT(i:i+step-1,:) = ii;
%         ii = ii-2;       
%     end
%     sigmaT = sigmaT/100;
%     sigmaT = sigmaT(1:sigmaT_resolution, 1:sigmaT_resolution);
%     sigmaT = imrotate(sigmaT,90);

%%
sigmaT = zeros(round(sqrt(2)*sigmaT_resolution),round(sqrt(2)*sigmaT_resolution));
ii = 100;
step = 1;
for i = 1:step:size(sigmaT,1)
   if(ii<0)
    ii = 100;
   end
    sigmaT(i:i+step-1,:) = ii;
    ii = ii-2;       
end
sigmaT = sigmaT/100;
sigmaT = imrotate(sigmaT,45);
startP = round((size(sigmaT,1)-sigmaT_resolution)/2);
sigmaT = sigmaT(startP:startP+sigmaT_resolution-1,startP:startP+sigmaT_resolution-1);

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
dlmwrite('input/sigmaT_diagonal_sparse.csv',sigmaT,'delimiter', ',', 'precision', 15);