import numpy as np
from computeFFT import computeFFT
import matplotlib.pyplot as plt
        
input = np.tile(0,(300,100))

output = computeFFT(input);
print(output.shape)

plt.plot(output[1,:],output[0,:])