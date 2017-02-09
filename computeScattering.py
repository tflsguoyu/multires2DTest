def computeScattering(hw_origin,hw_resize,albedo,NoSamples,receiptorSize,platform,numOfBlock):

    sigmaT_d_filename = 'output/sigmaTDownSample.csv';

    h_sigmaT_d = hw_resize[0]; w_sigmaT_d = hw_resize[1];
    h_region = 1; 
    w_region = h_region * (hw_origin[1] / hw_origin[0]);
                          
    if receiptorSize == 'MAX':
        receiptorSize = w_region;
  
    albedo_filename = 'output/albedo.csv';
    np.savetxt(albedo_filename, albedo, delimiter=',');

    
    if platform == 'Windows_C':
    # C++ windows
        system(['scatter.exe ' sigmaT_d_filename ' ' ...
            num2str(NoSamples) ' ' num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' '...
            num2str(h_region) ' ' num2str(w_region) ' ' num2str(receiptorSize) ' '...
            num2str(numOfBlock) ' ' albedo_filename]);

    if platform == 'Windows_C_nextEvent':
    # C++ windows
        system(['scatter_nextEvent.exe ' sigmaT_d_filename ' '...
            num2str(NoSamples) ' ' num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' '...
            num2str(h_region) ' ' num2str(w_region) ' ' num2str(receiptorSize) ' '...
            num2str(numOfBlock) ' ' albedo_filename]);
        
    if platform == 'Linux_C':
    # C++ Linux
        system(['./scatter_linux ' sigmaT_d_filename ' ' ...
            num2str(NoSamples) ' ' num2str(h_sigmaT_d) ' ' num2str(w_sigmaT_d) ' '...
            num2str(h_region) ' ' num2str(w_region) ' ' num2str(receiptorSize) ' '...
            num2str(numOfBlock) ' ' albedo_filename]);
    
#    if exist('output/densityMap.csv', 'file') == 2 :
#        output_insideVis = csvread('output/densityMap.csv');
#    else
    output_insideVis = np.zeros(1,1);

    output_reflection = np.loadtxt('output/reflectance.csv', delimiter=',');
    output_reflection_stderr = np.loadtxt('output/reflectanceStderr.csv', delimiter=',');
    
    # remove boundary block
    output_reflection[0,0] = sum(output_reflection[2:-2]);
    
    return (output_reflection,output_reflection_stderr,output_insideVis);
    