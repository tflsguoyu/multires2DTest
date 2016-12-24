function [albedo_adjust, downScale, std_d] = func_2DTest(sigmaT_inputFilename,tile, scale, albedo, ifDrawFFT, platform)
%    func_2DTest(sigmaT_inputFilename,tile, scale, albedo, ifDrawFFT, platform)


    %% Load SigmaT
    ext = sigmaT_inputFilename(end-2:end);
    if strcmp(ext,'csv')
        sigmaT = csvread(sigmaT_inputFilename);  
    end
    if strcmp(ext,'png')
        sigmaT = imread(sigmaT_inputFilename);
        if ndims(sigmaT) == 2
            sigmaT = im2double(sigmaT); 
        else
            sigmaT = im2double(rgb2gray(sigmaT));
        end
    end
    
    if tile
        sigmaT = repmat(sigmaT,[1 tile]);
    end
    
    sigmaT = scale * sigmaT;
    [h_sigmaT, w_sigmaT] = size(sigmaT);
    size_sigmaT = min(h_sigmaT, w_sigmaT);
    maxScale = ceil(log2(size_sigmaT));
    
    
    %% down sample
    for i = 0: maxScale
        downScale(i+1) = 2.^i;
    end
    N_downScale = length(downScale);
    mean_d = NaN(1,N_downScale);
    std_d = NaN(1,N_downScale);
    albedo_adjust = NaN(1,N_downScale);
    reflection = NaN(1,N_downScale);
    
    flag = 0;
    for windowsize = downScale
        
        flag = flag + 1;
        disp(['downsample: ' num2str(flag)]);
        
        %% down sampling sigmaT    
        sigmaT_d = imresize(imresize(sigmaT,1/windowsize,'box'),windowsize,'box');
%         sigmaT_d = imresize(sigmaT,1/windowsize,'box');
        [h,w] = size(sigmaT_d);
        disp(['sigmaT: ' num2str(h) ' x ' num2str(w)]);
        dlmwrite('output/sigmaTDownSample.csv', sigmaT_d, 'delimiter', ',', 'precision', 15);
        
        %% std of sigmaT
        mean_d(flag) = mean(sigmaT_d(:));
        std_d(flag) = std(sigmaT_d(:));

        if ifDrawFFT == 0
            subplot(2,N_downScale,flag);       
            imagesc(sigmaT_d);colormap(copper);
            axis off
            axis image
            h = colorbar('southoutside');
            t = get(h,'Limits');
            set(h,'Ticks',linspace(t(1),t(2),2));
%             axis equal
            title({['mean:' num2str(mean_d(flag))];['std:' num2str(std_d(flag))]});
        end
                
        %% fft of sigmaT
        if ifDrawFFT == 0      
            
            if size(sigmaT_d,1) > 2
                
                if size(sigmaT_d,1) < size(sigmaT_d,2)
                   sigmaT_d_cube = sigmaT_d(:, 1:size(sigmaT_d,1));
                else
                    sigmaT_d_cube = sigmaT_d;
                end

                [fft_log_NN, fft_window_list, fft_Ratio_list] = computeFFT(sigmaT_d_cube);

                subplot(4,N_downScale,flag+N_downScale*2)
                imagesc(fft_log_NN);
                axis off
                axis image
                title(['FFT'])
                
                subplot(4,N_downScale,flag+N_downScale*3)
                plot(fft_Ratio_list, fft_window_list, '-');
                xlabel('Ratio');
                ylabel('Window Size');
                axis equal
                axis([0 1 0 1])
            
            end
        
        end
        
        %% scattering    
        if ifDrawFFT == 1 || ifDrawFFT == 3
            sigmaT_filename = 'output/sigmaTDownSample.csv';
            N = 1000000;
            if flag == 1
    %             albedo = 0.95;

                sigmaT_d = csvread(sigmaT_filename);  
                [h_sigmaT_d,w_sigmaT_d] = size(sigmaT_d);
                h_region = 1;
                w_region = h_region * (w_sigmaT_d/h_sigmaT_d);
                
%                 tic;
                if strcmp(platform,'MATLAB')
                % MATLAB 
                    computeDensityMap(sigmaT_filename,albedo,N,...
                        h_sigmaT_d,w_sigmaT_d,h_region,w_region);
                end
                if strcmp(platform,'Windows_C')
                % C++ windows
                    system(['scatter.exe ' sigmaT_filename ' ' num2str(albedo) ' ' num2str(N) ' ' ...
                        num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
                end
                if strcmp(platform,'Linux_C')
                % C++ Linux
                    system(['./scatter_linux ' sigmaT_filename ' ' num2str(albedo) ' ' num2str(N) ' ' ...
                        num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
                end
%                 toc
                densityMap = csvread('output/densityMap.csv');
                reflection(flag) = csvread('output/reflectance.csv');
                albedo_adjust(flag) = albedo;
                [h,w] = size(densityMap);
                disp(['density map: ' num2str(h) ' x ' num2str(w)]);

            else
                albedo_start = 0;
                albedo_end = albedo+0.5;

                while 1
                    albedo_tmp = (albedo_start+albedo_end)/2;

                    sigmaT_d = csvread(sigmaT_filename);  
                    [h_sigmaT_d,w_sigmaT_d] = size(sigmaT_d);
                    h_region = 1;
                    w_region = h_region * (w_sigmaT_d/h_sigmaT_d);
                    
%                     tic;
                    if strcmp(platform,'MATLAB')
                    % MATLAB 
                        computeDensityMap(sigmaT_filename,albedo_tmp,N,...
                            h_sigmaT_d,w_sigmaT_d,h_region,w_region);
                    end
                    if strcmp(platform,'Windows_C')
                    % C++ windows
                        system(['scatter.exe ' sigmaT_filename ' ' num2str(albedo_tmp) ' ' num2str(N) ' ' ...
                            num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
                    end
                    if strcmp(platform,'Linux_C')
                    % C++ Linux
                        system(['./scatter_linux ' sigmaT_filename ' ' num2str(albedo_tmp) ' ' num2str(N) ' ' ...
                            num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
                    end
%                     toc
                    reflection_tmp = csvread('output/reflectance.csv');

                    err = reflection_tmp - reflection(1);
                    if abs(err) < 0.0001 || (albedo_end - albedo_start) < 0.00001
                        break;
                    end

                    if err < 0 
                        albedo_start = albedo_tmp;
                    else
                        albedo_end = albedo_tmp;
                    end           
                end

                reflection(flag) = reflection_tmp;
                albedo_adjust(flag) = albedo_tmp;
                densityMap = csvread('output/densityMap.csv');
                [h,w] = size(densityMap);
                disp(['density map: ' num2str(h) ' x ' num2str(w)]);
            end
            
        end
        
        %% display densityMap
        if ifDrawFFT == 3
            densityMap = log(densityMap);

            figure(222);
            subplot(N_downScale,1,flag)
            imagesc(densityMap)
            axis equal
            axis off      
            title(['r:' num2str(reflection(flag)) ' a:' num2str(albedo_adjust(flag))])
        end
    end
    
end

function computeDensityMap(filename_sigmaT_D,albedo,N_Sample,h_sigmaT_d,w_sigmaT_d,h,w)

    sigmaT_d_NN = csvread(filename_sigmaT_D);  
%     h_sigmaT_d = size(sigmaT_d_NN,1);
%     w_sigmaT_d = size(sigmaT_d_NN,2);
    
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
                    break;
                else
                    break;
                end
            elseif x(1) < 0.0 || x(1) > w || x(2) < 0.0
                break;
            end
            
            theta = 2*pi*rand;
            d = [cos(theta),sin(theta)];

            [r,c] = getCoord(x(1)/w,x(2)/h,h_sigmaT_d,w_sigmaT_d);
            [row,col] = getCoord(x(1)/w,x(2)/h,h_mapSize,w_mapSize);
    
            sigmaT = sigmaT_d_NN(r,c);
            densityMap(row,col) = densityMap(row,col) + weight/sigmaT;

            weight = weight * albedo;
        end
        
    end
    
    dlmwrite('output/reflectance.csv',reflectance,'delimiter', ',', 'precision', 15);
    dlmwrite('output/densityMap.csv',densityMap,'delimiter', ',', 'precision', 15);
    
end

function computeDensityMap_old(filename_sigmaT_D,albedo,N_Sample)

    sigmaT_d_NN = csvread(filename_sigmaT_D);  
    h_sigmaT_d = size(sigmaT_d_NN,1);
    w_sigmaT_d = size(sigmaT_d_NN,2);
        
    mapSize = 32;
    reflectance = 0;
    densityMap = zeros(mapSize,mapSize);
    
    for samples = 1: N_Sample

        maxDepth = 1000;        
        x = [rand,1];
        w = [0,1];       

%         [r,c] = getCoord(x(1),x(2),h_sigmaT_d,w_sigmaT_d);
        weight = 1/N_Sample;
         
        for dep = 1 : maxDepth
     
            [r,c] = getCoord(x(1),x(2),h_sigmaT_d,w_sigmaT_d);
            [row,col] = getCoord(x(1),x(2),mapSize,mapSize);
    
            sigmaT = sigmaT_d_NN(r,c);
            densityMap(row,col) = densityMap(row,col) + weight/sigmaT;
%             densityMap(row,col) = densityMap(row,col) + weight;
            
            
            %% method 2: Woodcock
            t = 0;
            while 1
                t = t - log(rand)/sigmaT;
                x_next = x - t * w;
                if x_next(1)<0 || x_next(1)>1 || x_next(2)<0 || x_next(2)>1
                    break;
                end
                [r_next,c_next] = getCoord(x_next(1),x_next(2),h_sigmaT_d,w_sigmaT_d);
                sigmaT_next = sigmaT_d_NN(r_next,c_next);
                if (sigmaT_next/sigmaT)>rand
                   break; 
                end
            end
            
            %% method 1: 
%             t = -log(rand)/sigmaT;
            %%
            x = x - t * w;
            
            if x(2) > 1.0
                intersectP_x = x(1) + (1-x(2))*w(1)/w(2);
                if intersectP_x > 0 && intersectP_x < 1
                    reflectance = reflectance + weight;
%                     reflectance = reflectance + weight/sigmaT;
                    break;
                else
                    break;
                end
            elseif x(1) < 0.0 || x(1) > 1.0 || x(2) < 0.0
                break;
            end
            
            theta = 2*pi*rand;
            w = [cos(theta),sin(theta)];


            weight = weight * albedo;
        end
        
    end
    
    csvwrite('output/reflectance.csv',reflectance);
    csvwrite('output/densityMap.csv',densityMap);
    
end


function [r,c] = getCoord(x,y,H,W)
    r = ceil((1-y)*H);
    c = ceil(x*W);
    r(r==0)=1;c(c==0)=1;
end

function [fft_log_NN,x_list,R_list] = computeFFT(img_NN)

    fft_NN = abs(fftshift(fft2(img_NN)));
    fft_log_NN = log(fft_NN+1);
    N = size(fft_NN,1);
        
    R_list = [0:0.01:1];
    x_list = [];
    for R = R_list
        x_start = 0;
        x_end = 1;
        err=1;
        while(err>0.0001 && (x_end-x_start)>0.0001)

            x = (x_start+x_end)/2;
            M = N*x;

            fft_sub_MM = fft_NN(round(N*(1-x)/2):round(N*(1+x)/2)+1, round(N*(1-x)/2):round(N*(1+x)/2)+1); 

            E_all = sum(fft_NN(:));
            E_sub = sum(fft_sub_MM(:));
            ratio = E_sub/E_all;

            if (ratio<R)
                x_start = x;
            else
                x_end = x;
            end

            err = abs(ratio-R);

        end
        x_list = [x_list,x];
    end
end








