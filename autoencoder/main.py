from autoencoder import Autoencoder
from sklearn import datasets

hidden_dimension = 1
data = datasets.load_iris().data
input_dimension = len(data[0])

autoencoder = Autoencoder(input_dimension, hidden_dimension)
autoencoder.train(data, learning_rate=0.001, epochs=100)

test_data = [[8, 4, 6, 2]]
print(test_data)
print(autoencoder.test(test_data))

