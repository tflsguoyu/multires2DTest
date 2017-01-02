function output = tileSigmaT(input, flag, tile)
    
    if strcmp(flag, 'x') 
        if tile
            output = repmat(input,[1 tile]);
        else
            output = input;    
        end
    end
    
end