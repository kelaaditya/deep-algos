import numpy as np
import os
import glob
from PIL import Image, ImageOps
from tflearn.data_utils import to_categorical

def load_data(path, size=(224, 224)):
    train_data_path_list = sorted(glob.glob(os.path.join(path, 'train/*'))[:100])
    X_train = np.zeros((len(train_data_path_list), *size, 3), dtype='float32')
    y_train = np.zeros(len(train_data_path_list))
    for i, pic in enumerate(train_data_path_list):
        im = Image.open(pic)
        #histogram equalization applied
        im = ImageOps.equalize(im)
        im = im.resize(size, Image.ANTIALIAS)
        X_train[i] = np.array(im)
        if 'dog' in pic:
            y_train[i] = 1
        else:
            y_train[i] = 0
    y_train = to_categorical(y_train, 2)
            
    test_data_path_list = sorted(glob.glob(os.path.join(path, 'test/*'))[:100])
    X_test = np.zeros((len(test_data_path_list), *size, 3), dtype='float32')
    for i, pic in enumerate(test_data_path_list):
        im = Image.open(pic)
        #histogram equalization applied
        im = ImageOps.equalize(im)
        im = im.resize(size, Image.ANTIALIAS)
        X_test[i] = np.array(im)
    
    return(X_train, y_train, X_test)