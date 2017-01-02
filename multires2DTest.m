function [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
    mean_d_list, std_d_list, reflection_list, insideVis_list, albedo_list] = ...
    multires2DTest(sigmaT_filename, scale, tile, max_downscale, albedo, platform, optimize)
% [downscale_list, sigmaT_d_list, logfft_d_list, fftcurve_d_list, ...
%    mean_d_list, std_d_list, reflection_list, insideVis_list, albedo_list] = ...
%    multires2DTest(sigmaT_filename, scale, tile, max_downscale, albedo, platform, optimize)

    format long;
    %% 
    sigmaT = loadSigmaT(sigmaT_filename);
    sigmaT = scaleSigmaT(sigmaT, scale);
    sigmaT = tileSigmaT(sigmaT, 'x', tile);    
    downscale_list = getDownscaleList(sigmaT, max_downscale);
      
    %%
    for flag = 1: length(downscale_list)
        disp(['downsample: ' num2str(flag)]);
        
        sigmaT_d = computeDownsampledSigmaT(sigmaT, downscale_list(flag));
        sigmaT_d_list{flag} = sigmaT_d;
        
        [logfft_d, fftcurve_d, mean_d, std_d] = computeFFT(sigmaT_d);  
        logfft_d_list{flag} = logfft_d;
        fftcurve_d_list(:,:,flag) = fftcurve_d;
        mean_d_list(flag) = mean_d;
        std_d_list(flag) = std_d;
                        
        % binary search    
        albedo_start = albedo-0.5;
        albedo_end = albedo+0.5;
        err = 1;
        iter = 0;
        while abs(err) > 0.0001 && (albedo_end - albedo_start) > 0.00001
            iter = iter + 1;
            
            albedo_tmp = (albedo_start+albedo_end)/2;
            [reflection,insideVis] = computeScattering(sigmaT_d,albedo_tmp,platform);

            if iter == 1
               reflection_iter1 = reflection;
               insideVis_iter1 = insideVis;
            end
            
            if strcmp(optimize,'no') || flag == 1
               break;
            end

            err = reflection - reflection_list(1);
            if err < 0 
                albedo_start = albedo_tmp;
            else
                albedo_end = albedo_tmp;
            end           
        end
        
        reflection_list(flag) = reflection_iter1;
        insideVis_list{flag} = insideVis_iter1;
        albedo_list(flag) = albedo_tmp;
       
    end
                
end








