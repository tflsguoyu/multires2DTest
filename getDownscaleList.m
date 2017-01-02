function output = getDownscaleList(input, max_downscale)
    
    if strcmp(max_downscale, 'MAX');
        [h, w] = size(input);
        max_downscale = ceil(log2(min(h, w)));
    end
    
    output = zeros(1,max_downscale+1);
    for i = 0: max_downscale
        output(i+1) = 2.^i;
    end

end