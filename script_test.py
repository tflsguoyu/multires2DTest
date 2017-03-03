import numpy as np
from multires2DTest_functions import getDownscaleList,computeDownsampledSigmaT,upsample
import matplotlib.pyplot as plt
        
downscale = [0,2]
downscale_list = getDownscaleList(downscale)
print(downscale_list)
downscaleTimes = np.size(downscale_list)
print(downscaleTimes)

sigmaT = np.mat(np.tile([0,0,0,0,0,0,0,0,1,1],4))
print(sigmaT.shape)
sigmaT_d = computeDownsampledSigmaT(sigmaT, downscale_list[1], 'x_average')
print(sigmaT_d.shape)
sigmaT_d = upsample(sigmaT_d, downscale_list[1], 'x_average')

print(sigmaT_d.shape)