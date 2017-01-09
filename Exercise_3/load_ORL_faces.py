import numpy as np  
 

# load data
data = np.load('ORL_faces.npz')
trainX = np.reshape(data['trainX'], (240, 1, 92, 112))/256
trainY = data['trainY']
testX = np.reshape(data['testX'], (160, 1, 92, 112))/256
testY = data['testY']
 


 
