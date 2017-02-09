import numpy as np

def tileSigmaT(input, flag, tile):
    
    if flag == 'x': 
        if tile:
            output = np.tile(input, (1,tile));
        else:
            output = input;    
        
    return output;