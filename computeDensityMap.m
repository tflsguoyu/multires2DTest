function computeDensityMap(filename_sigmaT_D,albedo,N_Sample,h_sigmaT_d,w_sigmaT_d,h,w)

    sigmaT_d_NN = csvread(filename_sigmaT_D);  
    
    sigmaT_MAX = max(sigmaT_d_NN(:));
    
    h_mapSize = 32;
    w_mapSize = h_mapSize * round(w/h);
    reflectance = 0;
    densityMap = zeros(h_mapSize,w_mapSize);
    
    for samples = 1: N_Sample

        maxDepth = 1000;        
        x = [rand*w,h];
        d = [0,1];       

        weight = 1/N_Sample;
         
        for dep = 1 : maxDepth
            
            %% method 2: Woodcock
            t = 0;
            while 1
                t = t - log(rand)/sigmaT_MAX;
                x_next = x - t * d;
                if x_next(1)<0 || x_next(1)>w || x_next(2)<0 || x_next(2)>h
                    break;
                end
                [r_next,c_next] = getCoord(x_next(1)/w,x_next(2)/h,h_sigmaT_d,w_sigmaT_d);
                sigmaT_next = sigmaT_d_NN(r_next,c_next);
                if (sigmaT_next/sigmaT_MAX)>rand
                   break; 
                end
            end
            
            %% method 1: 
%             t = -log(rand)/sigmaT;
            %%
            x = x - t * d;
            
            if x(2) > h
                intersectP_x = x(1) + (h-x(2))*d(1)/d(2);
                if intersectP_x > 0 && intersectP_x < w
                    reflectance = reflectance + weight;
                end
                break;
            elseif x(1) < 0.0 || x(1) > w || x(2) < 0.0
                break;
            end
            
            theta = 2*pi*rand;
            d = [cos(theta),sin(theta)];

            [r,c] = getCoord(x(1)/w,x(2)/h,h_sigmaT_d,w_sigmaT_d);
            [row,col] = getCoord(x(1)/w,x(2)/h,h_mapSize,w_mapSize);
    
            sigmaT = sigmaT_d_NN(r,c);
            densityMap(row,col) = densityMap(row,col) + weight/sigmaT;
            
            if dep <= 10
                weight = weight * albedo;
            else
                if rand > albedo
                    break;
                end
            end
        end
        
    end
    
    dlmwrite('output/reflectance.csv',reflectance,'delimiter', ',', 'precision', 16);
    dlmwrite('output/densityMap.csv',densityMap,'delimiter', ',', 'precision', 16);
    
end

% function computeDensityMap_old(filename_sigmaT_D,albedo,N_Sample)
% 
%     sigmaT_d_NN = csvread(filename_sigmaT_D);  
%     h_sigmaT_d = size(sigmaT_d_NN,1);
%     w_sigmaT_d = size(sigmaT_d_NN,2);
%         
%     mapSize = 32;
%     reflectance = 0;
%     densityMap = zeros(mapSize,mapSize);
%     
%     for samples = 1: N_Sample
% 
%         maxDepth = 1000;        
%         x = [rand,1];
%         w = [0,1];       
% 
% %         [r,c] = getCoord(x(1),x(2),h_sigmaT_d,w_sigmaT_d);
%         weight = 1/N_Sample;
%          
%         for dep = 1 : maxDepth
%      
%             [r,c] = getCoord(x(1),x(2),h_sigmaT_d,w_sigmaT_d);
%             [row,col] = getCoord(x(1),x(2),mapSize,mapSize);
%     
%             sigmaT = sigmaT_d_NN(r,c);
%             densityMap(row,col) = densityMap(row,col) + weight/sigmaT;
% %             densityMap(row,col) = densityMap(row,col) + weight;
%             
%             
%             %% method 2: Woodcock
%             t = 0;
%             while 1
%                 t = t - log(rand)/sigmaT;
%                 x_next = x - t * w;
%                 if x_next(1)<0 || x_next(1)>1 || x_next(2)<0 || x_next(2)>1
%                     break;
%                 end
%                 [r_next,c_next] = getCoord(x_next(1),x_next(2),h_sigmaT_d,w_sigmaT_d);
%                 sigmaT_next = sigmaT_d_NN(r_next,c_next);
%                 if (sigmaT_next/sigmaT)>rand
%                    break; 
%                 end
%             end
%             
%             %% method 1: 
% %             t = -log(rand)/sigmaT;
%             %%
%             x = x - t * w;
%             
%             if x(2) > 1.0
%                 intersectP_x = x(1) + (1-x(2))*w(1)/w(2);
%                 if intersectP_x > 0 && intersectP_x < 1
%                     reflectance = reflectance + weight;
% %                     reflectance = reflectance + weight/sigmaT;
%                     break;
%                 else
%                     break;
%                 end
%             elseif x(1) < 0.0 || x(1) > 1.0 || x(2) < 0.0
%                 break;
%             end
%             
%             theta = 2*pi*rand;
%             w = [cos(theta),sin(theta)];
% 
% 
%             weight = weight * albedo;
%         end
%         
%     end
%     
%     csvwrite('output/reflectance.csv',reflectance);
%     csvwrite('output/densityMap.csv',densityMap);
%     
% end


function [r,c] = getCoord(x,y,H,W)
    r = ceil((1-y)*H);
    c = ceil(x*W);
    r(r==0)=1;c(c==0)=1;
end


