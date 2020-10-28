'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause

The purpose of this module is to demonstrate the process of obtaining pixel-wise explanations for given data points at hand of the MNIST hand written digit data set.

The module first loads a pre-trained neural network model and the MNIST test set with labels and transforms the data such that each pixel value is within the range of [-1 1].
The data is then randomly permuted and for the first 10 samples due to the permuted order, a prediction is computed by the network, which is then as a next step explained
by attributing relevance values to each of the input pixels.

finally, the resulting heatmap is rendered as an image and (over)written out to disk and displayed.
'''


import matplotlib.pyplot as plt
import time
import numpy
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
na = np.newaxis

import model_io
import data_io
import render

def add_salt_and_pepper_noise(prob, image):
        
    #Add salt and pepper noise to image
    #prob: Probability of the noise
    
    
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
        
    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    
    
    return image


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

#load a neural network, as well as the MNIST test data and some labels
nn = model_io.read('../mnist_mlp_big_tanh-1296-1296-1296.txt') # 99.16% prediction accuracy
nn.drop_softmax_output_layer() #drop softnax output layer for analyses

X = data_io.read('../data/MNIST/test_images.npy')
Y = data_io.read('../data/MNIST/test_labels.npy')

print("look at me")
print(X.shape)

prob =0.10

X = add_salt_and_pepper_noise(prob, X)
    
X_original = data_io.read('../data/MNIST/test_images.npy')    
    
X = create_preprocessing(X, input_range=[-1, 1])
    

X_original = np.reshape(X_original,[X_original.shape[0],28,28,1])
X_original = np.pad(X_original,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))    
    

    
#reshape the vector representations of the mnist data back to image format. extend the image vertically and horizontally by 4 pixels each.


X = np.reshape(X,[X.shape[0],28,28,1])
X = np.pad(X,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1

acc = np.mean(np.argmax(nn.forward(X), axis=1) == np.argmax(Y, axis=1))
if not np == numpy: # np=cupy
    acc = np.asnumpy(acc)
print('model test accuracy is: {:0.4f}'.format(acc))

#permute data order for demonstration. or not. your choice.
I = np.arange(X.shape[0])
#I = np.random.permutation(I)



show_digit_a = 4
show_digit_b = 3

show_digit_a_count = 0
show_digit_b_count = 0

show_pred = [4,9,3,8]
#show_pred = [0,1,2,3,4,5,6,7,8,9]

show_count = 8
counting = 0

#predict and perform LRP for the 10 first samples
for i in I[:100]:
    x = X[na,i,:]
    
    counting+=1
    
    x_original =X_original[na,i,:]

    #forward pass and prediction
    ypred = nn.forward(x)
    #print('True Class:     ', np.argmax(Y[i]))
    #print('Predicted Class:', np.argmax(ypred),'\n')
    
    #if counting==17:
        #break
    
    #'''
    if (np.argmax(ypred)==np.argmax(Y[i])) and (np.argmax(ypred)==show_digit_a  or np.argmax(ypred)==show_digit_b):
        
        #print(np.argmax(ypred))
        
        if np.argmax(ypred)==show_digit_a:
            show_digit_a_count+=1
        else:
            show_digit_b_count+=1
            
        if  show_digit_a_count==show_count and  show_digit_a_count==show_count:
            break
    
    #'''

        #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
        
        #mask[:,np.argmax(ypred)] = 1
        
        for j in show_pred:
            
            mask = np.zeros_like(ypred)
            
            mask[:,j] = 1
        
        
        
            Rinit = ypred*mask

            #compute first layer relevance according to prediction
            R = nn.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
            #R = nn.lrp(Rinit,'epsilon',0.01)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
            #R = nn.lrp(Rinit,'alphabeta',2)    #as Eq(60) from DOI: 10.1371/journal.pone.0130140


            #R = nn.lrp(ypred*Y[na,i]) #compute first layer relevance according to the true class label
            '''
            yselect = 3
            yselect = (np.arange(Y.shape[1])[na,:] == yselect)*1.
            R = nn.lrp(ypred*yselect) #compute first layer relvance for an arbitrarily selected class
            '''

            #undo input normalization for digit drawing. get it back to range [0,1] per pixel
            #x = (x+1.)/2.

            if not np == numpy: # np=cupy
                x_original = np.asnumpy(x_original) #carefull
                R = np.asnumpy(R)

            #render input and heatmap as rgb images
            digit = render.digit_to_rgb(x_original, scaling = 3)
            hm = render.hm_to_rgb(R, X = x_original, scaling = 3, sigma = 2)
            #digit_hm = render.save_image([digit,hm],'../heatmap_'+str(i)+'_'+str(j)+'_'+str(show_digit_a_count)+'_'+str(np.argmax(ypred))+'.png')
            digit_hm = render.save_image([hm],'exp_2_tanh/'+str(np.argmax(ypred))+'_'+str(j)+'_heatmap_'+str(i)+'_'+str(show_digit_a_count)+'.png')
            #digit_hm = render.save_image([hm],'exp_1/'+str(np.argmax(ypred))+'_'+str(j)+'_heatmap_'+str(i)+'.png')
            digit_im = render.save_image([digit],'exp_2_tanh/'+str(i)+'_heatmap.png')
            data_io.write(R,'../heatmap.npy')

            #display the image as written to file
            #plt.imshow(digit_hm, interpolation = 'none')
            #plt.axis('off')
            #plt.show()


#note that modules.Sequential allows for batch processing inputs
'''
if True:
    N = 256
    t_start = time.time()
    x = X[:N,...]
    y = nn.forward(x)
    R = nn.lrp(y)
    data_io.write(R,'../Rbatch.npy')
    print('Computation of {} heatmaps using {} in {:.3f}s'.format(N, np.__name__, time.time() - t_start))
'''
