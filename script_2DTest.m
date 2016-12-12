function test
    clear
    close all
%     format long
   
    %% Load SigmaT
    sigmaT = csvread('input/sigmaT_binaryRand.csv');  
    scale = 10;
    sigmaT = scale * sigmaT;
    sigmaT_size = size(sigmaT,1);
    
    %% down sample
    downScale = [1,2,4,8,16,32];
    N_downScale = length(downScale);
    figure;
    flag = 0;
    for windowsize = downScale
        
        flag = flag + 1
        
        %% down sampling sigmaT    
        % method 1:
        sigmaT_d_NN = imresize(imresize(sigmaT,1/windowsize,'box'),windowsize,'box');
%         sigmaT_d_NN = imresize(imresize(sigmaT,1/windowsize),windowsize);
        

        csvwrite('output/sigmaTDownSample.csv', sigmaT_d_NN);
        
        %% std of sigmaT
        std_d(flag) = std(sigmaT_d_NN(:));
        
        subplot(3,N_downScale,flag);
        imagesc(sigmaT_d_NN, [0 6])
        colorbar
        axis equal
        axis off
        title(['std:' num2str(std_d(flag))]);
                
        %% fft of sigmaT
        [fft_log_NN, fft_window_list, fft_Ratio_list] = computeFFT(sigmaT_d_NN);
        diff_fft_window_list = diff(fft_window_list);
        idx = find(diff_fft_window_list);
        fft_Ratio(flag) = fft_Ratio_list(idx(1));
        fft_window(flag) = fft_window_list(fft_Ratio_list==0.9);
        N = size(fft_log_NN,1);
        
%         subplot(3,N_downScale,flag+N_downScale)
%         imagesc(fft_log_NN);
%         colorbar;hold on;
%         rectangle('Position',[N*(1-fft_x(flag))/2 N*(1-fft_x(flag))/2 N*fft_x(flag)+1 N*fft_x(flag)+1],'EdgeColor','w');hold off;
%         axis equal
%         axis off
%         title(['fft:' num2str(fft_x(flag))])

        subplot(3,N_downScale,flag+N_downScale)
        plot(fft_Ratio_list, fft_window_list, '+-');
        xlabel('Ratio');
        ylabel('windowSize');
        
        %% scattering    
        sigmaT_filename = 'output/sigmaTDownSample.csv';
        N = 10000000;
        if flag == 1
            albedo = 0.95;
        
            % MATLAB 
    %         computeDensityMap(sigmaT_filename,albedo,N);
            % C++ windows
%             system(['scatter.exe ' sigmaT_filename ' ' num2str(albedo) ' ' num2str(N)]);
            % C++ Linux
            system(['./scatter_linux ' sigmaT_filename ' ' num2str(albedo) ' ' num2str(N)]);
            densityMap = csvread('output/densityMap.csv');
            reflection(flag) = csvread('output/reflectance.csv');
            albedo_adjust(flag) = albedo;

        else
            albedo_start = albedo-0.5;
            albedo_end = albedo;
            
            while 1
                albedo_tmp = (albedo_start+albedo_end)/2;
                
                % MATLAB 
        %         computeDensityMap(sigmaT_filename,albedo,N);
                % C++ windows
%                 system(['scatter.exe ' sigmaT_filename ' ' num2str(albedo_tmp) ' ' num2str(N)]);
                % C++ Linux
                system(['./scatter_linux ' sigmaT_filename ' ' num2str(albedo_tmp) ' ' num2str(N)]);
                reflection_tmp = csvread('output/reflectance.csv');

                err = reflection_tmp - reflection(1);
                if abs(err) < 0.0001 || (albedo_end - albedo_start) < 0.0000001
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
            
        end
            
        
        %% display densityMap
        densityMap = log(densityMap);
%         densityMean(flag) = sum(densityMap(:));

        subplot(3,N_downScale,flag+N_downScale*2)
        imagesc(densityMap, [-10 -5])
        colorbar
        axis equal
        axis off      
        title(['r:' num2str(reflection(flag)) ' a:' num2str(albedo_adjust(flag))])
    end
    
    %% Draw curve
    figure;

    subplot(2,2,1);
    plot(log2(downScale),albedo_adjust,'*-');
    xlabel('log downsampleScale');
    ylabel('albedo');

    subplot(2,2,2);
    plot(std_d,albedo_adjust,'*-');
    xlabel('std');
    ylabel('albedo');

    subplot(2,2,3);
    plot(fft_window,albedo_adjust,'*-');
    xlabel('fft windowsize');
    ylabel('albedo');
    
    subplot(2,2,4);
    plot(fft_Ratio,albedo_adjust,'*-');
    xlabel('fft ratio');
    ylabel('albedo');
    
end

function computeDensityMap(filename_sigmaT_D,albedo,N_Sample)

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
    
    R_list = [0:0.02:1];
    x_list = [];
    for R = R_list
        x_start = 0;
        x_end = 1;
        err=1;
        while(err>0.001 && (x_end-x_start)>0.001)

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








