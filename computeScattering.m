function [output_reflection,output_reflection_stderr,output_insideVis] ...
    = computeScattering(hw_origin,hw_resize,albedo,NoSamples,platform)

    sigmaT_d_filename = 'output/sigmaTDownSample.csv';

    h_sigmaT_d = hw_resize(1); w_sigmaT_d = hw_resize(2);
    h_region = 1; w_region = h_region * (hw_origin(2)/hw_origin(1));

    if strcmp(platform,'MATLAB')
    % MATLAB 
        scatter(sigmaT_d_filename,albedo,NoSamples,...
            h_sigmaT_d,w_sigmaT_d,h_region,w_region);
    end
    
    if strcmp(platform,'Windows_C')
    % C++ windows
        system(['scatter.exe ' sigmaT_d_filename ' ' num2str(albedo) ' ' num2str(NoSamples) ' ' ...
            num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
    end
    
    if strcmp(platform,'Linux_C')
    % C++ Linux
        system(['./scatter_linux ' sigmaT_d_filename ' ' num2str(albedo) ' ' num2str(NoSamples) ' ' ...
            num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
    end
    
    if exist('output/densityMap.csv', 'file') == 2
        output_insideVis = csvread('output/densityMap.csv');
    else
        output_insideVis = zeros(1);
    end
    output_reflection = csvread('output/reflectance.csv');
    output_reflection_stderr = csvread('output/reflectanceStderr.csv');
    

end