function output = loadSigmaT(filename)
    
    ext = filename(end-2:end);
    
    if strcmp(ext,'csv')
        output = csvread(filename);  
    end
    
    if strcmp(ext,'png') || strcmp(ext,'bmp')
        output = imread(filename);
        if ndims(output) == 3
            output = rgb2gray(output);
        end
        output = im2double(output); 
    end

end