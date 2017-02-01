function [output_reflection,output_reflection_stderr,output_insideVis] ...
    = computeScattering(hw_origin,hw_resize,albedo,NoSamples,receiptorSize,...
    platform,numOfBlock)

    sigmaT_d_filename = 'output/sigmaTDownSample.csv';

    h_sigmaT_d = hw_resize(1); w_sigmaT_d = hw_resize(2);
    h_region = 1; w_region = h_region * (hw_origin(2)/hw_origin(1));
    if strcmp(receiptorSize,'MAX')
        receiptorSize = w_region;
    end

    albedo_filename = 'output/albedo.csv';
    csvwrite(albedo_filename, albedo);
    
    if strcmp(platform,'MATLAB')
    % MATLAB 
        scatter(sigmaT_d_filename,albedo,NoSamples,...
            h_sigmaT_d,w_sigmaT_d,h_region,w_region,receiptorSize,numOfBlock);
    end
    
    if strcmp(platform,'Windows_C')
    % C++ windows
        system(['scatter.exe ' sigmaT_d_filename ' ' ...
            num2str(NoSamples) ' ' num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' '...
            num2str(h_region) ' ' num2str(w_region) ' ' num2str(receiptorSize) ' '...
            num2str(numOfBlock) ' ' albedo_filename]);
    end

    if strcmp(platform,'Windows_C_nextEvent')
    % C++ windows
        system(['scatter_nextEvent.exe ' sigmaT_d_filename ' '...
            num2str(NoSamples) ' ' num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' '...
            num2str(h_region) ' ' num2str(w_region) ' ' num2str(receiptorSize) ' '...
            num2str(numOfBlock) ' ' albedo_filename]);
    end
        
    if strcmp(platform,'Linux_C')
    % C++ Linux
        system(['./scatter_linux ' sigmaT_d_filename ' ' num2str(albedo) ' '...
            num2str(NoSamples) ' ' num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' '...
            num2str(h_region) ' ' num2str(w_region) ' ' num2str(receiptorSize) ' '...
            num2str(numOfBlock)]);
    end
    
    if exist('output/densityMap.csv', 'file') == 2
        output_insideVis = csvread('output/densityMap.csv');
    else
        output_insideVis = zeros(1);
    end
    output_reflection = csvread('output/reflectance.csv');
    output_reflection_stderr = csvread('output/reflectanceStderr.csv');
    
    % remove boundary block
    output_reflection(1) = sum(output_reflection(3:end-1));
    
    guoyu = 1;
    

end