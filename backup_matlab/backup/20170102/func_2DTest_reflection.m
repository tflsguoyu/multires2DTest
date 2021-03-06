function [reflection, downScale, std_d] = func_2DTest_reflection(sigmaT_inputFilename,tile, scale, albedo, ifDrawFFT, platform)
%    func_2DTest(sigmaT_inputFilename,tile, scale, albedo, ifDrawFFT, platform)

    format long;
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
    for i = 0 : 1%maxScale
        downScale(i+1) = 2.^i;
    end
    N_downScale = length(downScale);
    std_d = NaN(1,N_downScale);
    albedo_adjust = NaN(1,N_downScale);
    reflection = NaN(1,N_downScale);
    
    flag = 0;
    for windowsize = downScale
        
        flag = flag + 1;
        disp(['downsample: ' num2str(flag)]);
        
        %% down sampling sigmaT    
        sigmaT_d = imresize(imresize(sigmaT,1/windowsize,'box'),[h_sigmaT w_sigmaT],'box');
%         sigmaT_d = imresize(imresize(sigmaT,[h_sigmaT round(1/windowsize*w_sigmaT)],'box'),[h_sigmaT w_sigmaT],'box');
%         sigmaT_d = imresize(sigmaT,1/windowsize,'box');
        [h,w] = size(sigmaT_d);
        disp(['sigmaT: ' num2str(h) ' x ' num2str(w)]);
        dlmwrite('output/sigmaTDownSample.csv', sigmaT_d, 'delimiter', ',', 'precision', 16);
        
        %% std of sigmaT
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
            title(['std:' num2str(std_d(flag))]);
        end
                
        %% fft of sigmaT
        if ifDrawFFT == 0         
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
            
            diff_fft_window_list = diff(fft_window_list);
            idx = find(diff_fft_window_list);
            fft_Ratio(flag) = fft_Ratio_list(idx(1));
            fft_window(flag) = fft_window_list(fft_Ratio_list==0.9);
            N = size(fft_log_NN,1);

            subplot(4,N_downScale,flag+N_downScale*3)
            plot(fft_Ratio_list, fft_window_list, '-');
            xlabel('Ratio');
            ylabel('Window Size');
            axis equal
            axis([0 1 0 1])
        
        end
        
        %% scattering    
        if ifDrawFFT == 1 || ifDrawFFT == 3
            sigmaT_filename = 'output/sigmaTDownSample.csv';
            N = 1000000;

            sigmaT_d = csvread(sigmaT_filename);  
            [h_sigmaT_d,w_sigmaT_d] = size(sigmaT_d);
            h_region = 1;
            w_region = h_region * (w_sigmaT_d/h_sigmaT_d);

%             tic;
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
%             toc;
            reflection(flag) = csvread('output/reflectance.csv');  

            densityMap = csvread('output/densityMap.csv');
            [h,w] = size(densityMap);
            disp(['density map: ' num2str(h) ' x ' num2str(w)]);

            
        end
        
        %% display densityMap
        if ifDrawFFT == 3
            densityMap = log(densityMap);

            figure(222);
            subplot(N_downScale,1,flag)
            imagesc(densityMap)
            axis equal
            axis off      
            title(['r:' num2str(reflection(flag)) ' a:' num2str(albedo)])
        end
    end
    
end









