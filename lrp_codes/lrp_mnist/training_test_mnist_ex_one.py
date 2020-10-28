'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 30.09.2015
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import modules
import model_io
import data_io

import importlib.util as imp
import numpy
import numpy as np
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
    
    
    
    
def create_preprocessing(X, input_range=[0, 1]):
    """
    Generically shifts data from interval [a, b] to interval [c, d].
    Assumes that theoretical min and max values are populated.
    """

    if len(input_range) != 2:
        raise ValueError(
            "Input range must be of length 2, but was {}".format(
                len(input_range)))
    if input_range[0] >= input_range[1]:
        raise ValueError(
            "Values in input_range must be ascending. It is {}".format(
                input_range))

    a, b = X.min(), X.max()
    c, d = input_range


    # shift original data to [0, b-a] (and copy)
    X = X - a
    # scale to new range gap [0, d-c]
    X = X / (b-a)
    X *= (d-c)
    # shift to desired output range
    X += c
    return X

    
    
    
na = np.newaxis


train_mnist = True
salt_and_pepper_noise = True



if train_mnist:

    Xtrain = data_io.read('../data/MNIST/train_images.npy')
    Ytrain = data_io.read('../data/MNIST/train_labels.npy')
    Xtest = data_io.read('../data/MNIST/test_images.npy')
    Ytest = data_io.read('../data/MNIST/test_labels.npy')
    
    #print(Xtest.mean(axis=1))
    #print(Xtest.std(axis=1))

    # transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
    #Xtrain =  Xtrain / 127.5 - 1
    #Xtest =  Xtest / 127.5 - 1
    
    Xtrain = create_preprocessing(Xtrain, input_range=[0, 1])
    Xtest = create_preprocessing(Xtest, input_range=[0, 1])
    

    
    
    #print(np.sum(Xtest.mean(axis=1))/Xtest.shape[0])
    #print(np.sum(Xtest.std(axis=1))/Xtest.shape[0])
    
    #reshape the vector representations of the mnist data back to image format. extend the image vertically and horizontally by 4 pixels each.
    Xtrain = np.reshape(Xtrain,[Xtrain.shape[0],28,28,1])
    #Xtrain = np.pad(Xtrain,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

    Xtest = np.reshape(Xtest,[Xtest.shape[0],28,28,1])
    #Xtest = np.pad(Xtest,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))
    
    #print(Xtest.shape)
    
  

    # transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
    I = Ytrain[:,0].astype(int)
    Ytrain = np.zeros([Xtrain.shape[0],np.unique(Ytrain).size])
    Ytrain[np.arange(Ytrain.shape[0]),I] = 1
    
    I = Ytest[:,0].astype(int)
    Ytest = np.zeros([Xtest.shape[0],np.unique(Ytest).size])
    Ytest[np.arange(Ytest.shape[0]),I] = 1
    #'''
    nn = modules.Sequential(
        [
            modules.Flatten(),
            modules.Linear(784, 400),
            modules.Tanh(),
            modules.Linear(400,400),
            modules.Tanh(),
            modules.Linear(400,400),
            modules.Tanh(),
            modules.Linear(400, 10),
            modules.SoftMax()
        ]
    )
    
    nn.train(Xtrain, Ytrain, Xtest, Ytest, batchsize=25, iters=50000, status=1000, transform=True)
    acc = np.mean(np.argmax(nn.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
    if not np == numpy: # np=cupy
        acc = np.asnumpy(acc)
    print('model test accuracy is: {:0.4f}'.format(acc))
    model_io.write(nn, '../mnist_mlp-400-400-400.txt')
    #'''
    #try loading the model again and compute score, see if this checks out. this time in numpy
    nn = model_io.read('../mnist_mlp-400-400-400.txt')
    acc = np.mean(np.argmax(nn.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
    if not np == numpy: acc = np.asnumpy(acc)
    print('model test accuracy (numpy) is: {:0.4f}'.format(acc))
