import numpy as np

from keras.models import load_model

# build the VGG16 network
model = load_model('output/myModel.h5')
print('Model loaded.')

weight = model.get_weights()
print(np.shape(weight))

for i in range(len(weight)):
    weight_this = weight[i]
    np.savetxt('output/layer'+repr(i+1)+'.csv', weight_this, delimiter=',')
    
