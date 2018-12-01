import numpy as np
import os
import scipy.io
import tensorflow as tf


VGG19_LAYERS = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
        )


class VGG19:
    def __init__(self, path_to_vgg):
        self.path_to_vgg = path_to_vgg
        self.image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='image')
        
    def _conv_layer(self, input_layer, weights, bias):
        conv_layer = tf.nn.conv2d(input=input_layer,
                                  filter=tf.constant(weights, name='conv_weights'),
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')
        return tf.nn.bias_add(conv_layer, bias)
    
    def _relu_layer(self, input_layer):
        return tf.nn.relu(input_layer)
    
    def _pool_layer(self, input_layer, pool_func='max'):
        '''Pooling chosen between maximum or average
        '''
        
        if pool_func == 'max':
            return tf.nn.max_pool(value=input_layer,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID')
        else:
            return tf.nn.avg_pool(value=input_layer,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID')
        
    def build_graph(self):
        vgg = scipy.io.loadmat(self.path_to_vgg)
        vgg_layers = vgg['layers']
        current_layer = self.image
        
        for i, layer in enumerate(VGG19_LAYERS):
            layer_type = layer[:4]
            
            if layer_type == 'conv':
                weights, bias = vgg_layers[0][i][0][0][2][0]
                bias = bias.reshape(bias.size)
                current_layer = self._conv_layer(current_layer, weights, bias)
            elif layer_type == 'relu':
                current_layer = self._relu_layer(current_layer)
            elif layer_type == 'pool':
                current_layer = self._pool_layer(current_layer)
                
            if layer == 'conv5_3':
                self.features = tf.reshape(current_layer, [-1, 196, 512])


if __name__=="__main__":
    base_folder = os.path.dirname(__file__)
    path_to_vgg = os.path.join(base_folder, 'data', 'imagenet-vgg-verydeep-19.mat')
    encoder = VGG19(path_to_vgg)
    encoder.build_graph()

