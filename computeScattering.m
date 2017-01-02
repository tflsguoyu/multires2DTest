function [output_reflection,output_insideVis] = computeScattering(input,albedo,platform)

    sigmaT_d_filename = 'output/sigmaTDownSample.csv';
    N = 1000000;

    [h_sigmaT_d,w_sigmaT_d] = size(input);
    h_region = 1; w_region = h_region * (w_sigmaT_d/h_sigmaT_d);

    if strcmp(platform,'MATLAB')
    % MATLAB 
        scatter(sigmaT_d_filename,albedo,N,...
            h_sigmaT_d,w_sigmaT_d,h_region,w_region);
    end
    
    if strcmp(platform,'Windows_C')
    % C++ windows
        system(['scatter.exe ' sigmaT_d_filename ' ' num2str(albedo) ' ' num2str(N) ' ' ...
            num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
    end
    
    if strcmp(platform,'Linux_C')
    % C++ Linux
        system(['./scatter_linux ' sigmaT_d_filename ' ' num2str(albedo) ' ' num2str(N) ' ' ...
            num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' ' num2str(h_region) ' ' num2str(w_region)]);
    end
    
    output_insideVis = csvread('output/densityMap.csv');
    output_reflection = csvread('output/reflectance.csv');

end