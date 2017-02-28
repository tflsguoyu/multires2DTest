function [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
    mean_d_list, std_d_list, reflection_list, reflection_stderr_list, ...
    reflectionOptimize_list, insideVis_list, albedo_k_list] = ...
    multires2DTest(sigmaT_filename, scale, tile, max_downscale, albedo, NoSamples, receiptorSize, platform, optimize, numOfBlock)
% [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
%    mean_d_list, std_d_list, reflection_list, reflection_stderr_list, insideVis_list, albedo_list] = ...
%    multires2DTest(sigmaT_filename, scale, tile, max_downscale, albedo, N, platform, optimize)

    format long;
    %% 
    sigmaT = loadSigmaT(sigmaT_filename); [h_origin,w_origin] = size(sigmaT);
    sigmaT = scaleSigmaT(sigmaT, scale);
    sigmaT = tileSigmaT(sigmaT, 'x', tile); [h_tile,w_tile] = size(sigmaT);   
    downscale_list = getDownscaleList(sigmaT, max_downscale);
      
    %%
    for flag = 1: length(downscale_list)
        disp(['downsample: ' num2str(flag-1) '/' num2str(length(downscale_list)-1)]);
        
        sigmaT_d = computeDownsampledSigmaT(sigmaT, downscale_list(flag),'x_average'); 
        [h_resize,w_resize] = size(sigmaT_d);
        sigmaT_d = imresize(sigmaT_d, [h_tile,w_tile], 'box');
        sigmaT_d_list{flag} = sigmaT_d(1:h_origin,1:w_origin);
        
        [logfft_d, fftcurve_d, mean_d, std_d] = computeFFT(sigmaT_d);  
        logfft_d_list{flag} = imresize(logfft_d,[size(logfft_d,1) size(logfft_d,1)]);
        fftcurve_d_list(:,:,flag) = fftcurve_d;
        mean_d_list(flag) = mean_d;
        std_d_list(flag) = std_d;
                        
        % binary search  
        albedo_k_start = 0.5;
        albedo_k_end = 1.5;
        err = 1;
        iter = 0;
        while abs(err) > 0.0001 && (albedo_k_end - albedo_k_start) > 0.00001
            iter = iter + 1;
            
            albedo_k_tmp = (albedo_k_start+albedo_k_end)/2;
            
            [reflection,reflection_stderr,insideVis] = ...
                computeScattering([h_tile,w_tile],[h_resize,w_resize],...
                albedo_k_tmp*albedo,NoSamples,receiptorSize,platform,numOfBlock);

            if iter == 1
               reflection_iter1 = reflection;
               reflection_stderr_iter1 = reflection_stderr;
               insideVis_iter1 = insideVis;
            end
            
            if strcmp(optimize,'no') || flag == 1
               break;
            end

            err = reflection(1) - reflection_list(1,1);
            if err < 0 
                albedo_k_start = albedo_k_tmp;
            else
                albedo_k_end = albedo_k_tmp;
            end           
        end
        
        reflection_list(flag,:) = reflection_iter1;
        reflection_stderr_list(flag,:) = reflection_stderr_iter1;
        
        reflectionOptimize_list(flag,:) = reflection;

        insideVis_list{flag} = imresize(insideVis_iter1,[size(insideVis_iter1,1) size(insideVis_iter1,1)]);
        albedo_k_list(flag) = albedo_k_tmp;
       
        deleteTmpFiles();
    end
                
end









