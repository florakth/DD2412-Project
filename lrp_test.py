#!/usr/bin/env python
# coding: utf-8

# # Compare analyzers on ImageNet

# In this notebook we show how one can use **iNNvestigate** to analyze the prediction of ImageNet-models! To do so we will load a network from the keras.applications module and analyze prediction on some images!
# 
# Parts of the code that do not contribute to the main focus are outsourced into utility modules. To learn more about the basic usage of **iNNvestigate** have look into this notebook: [Introduction to iNNvestigate](introduction.ipynb) and [Comparing methods on MNIST](mnist_method_comparison.ipynb)
# 
# -----
# 
# **To use this notebook please download the example images using the following script:**
# 
# `innvestigate/examples/images/wget_imagenet_2011_samples.sh`

# ## Imports

# In[1]:


#%load_ext autoreload
#%autoreload 2

import warnings
warnings.simplefilter('ignore')


# In[ ]:


import os
import numpy as np
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score
import sklearn.metrics as metrics
import random

use_gpu = torch.cuda.is_available()


# In[2]:


#%matplotlib inline  

import tensorflow as tf

import imp
import numpy as np
import os

import tensorflow.keras as keras
import tensorflow.keras.backend 
import tensorflow.keras.models 

import innvestigate
import innvestigate.applications.imagenet
import innvestigate.utils as iutils

# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", "/home/dsv/maul8511/deep_learning_project/referance_codes/innvestigate/examples/utils.py")
imgnetutils = imp.load_source("utils_imagenet", "/home/dsv/maul8511/deep_learning_project/referance_codes/innvestigate/examples/utils_imagenet.py")


# ## Model

# In this demo use the VGG16-model, which uses ReLU activation layers.

# In[3]:


'''
# Load the model definition.
tmp = getattr(innvestigate.applications.imagenet, os.environ.get("NETWORKNAME", "vgg16"))
net = tmp(load_weights=True, load_patterns="relu")

# Build the model.
model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Handle input depending on model and backend.
channels_first = keras.backend.image_data_format() == "channels_first"
color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
'''


# In[ ]:


model = tf.keras.models.load_model('keras_model')
model.compile(optimizer="adam", loss="binary_crossentropy")
print("loaded successfully")

# Handle input depending on model and backend.
channels_first = keras.backend.image_data_format() == "channels_first"
color_conversion =  None


# In[ ]:


image_shape = [224, 224]

net = {}

net["image_shape"] = image_shape
    
if keras.backend.image_data_format() == "channels_first":
    net["input_shape"] = [None, 3]+image_shape
else:
    net["input_shape"] = [None]+image_shape+[3]
    
net["input_range"] = {
        None: (-128, 128),
        "caffe": (-128, 128),
        "tf": (-1, 1),
        "torch": (-3, 3),
    }["torch"]

preprocess_f=keras.applications.densenet.preprocess_input
net["preprocess_f"] = preprocess_f


# ## Data

# Now we load some example images and preprocess them to fit the input size model.
# 
# To analyze your own example images, just add them to `innvestigate/examples/images`.

# In[4]:


# Get some example test set images.
'''
images, label_to_class_name = eutils.get_imagenet_data(net["image_shape"][0])

if not len(images):
    raise Exception("Please download the example images using: "
                    "'innvestigate/examples/images/wget_imagenet_2011_samples.sh'")
                    
'''


# In[ ]:


# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
pathFileTrain = '/home/dsv/maul8511/deep_learning_project/data/CheXpert-v1.0-small/train.csv'
pathFileValid = '/home/dsv/maul8511/deep_learning_project/data/CheXpert-v1.0-small/valid.csv'

# Neural network parameters:
nnIsTrained = False                 #pre-trained using ImageNet
nnClassCount = 14                   #dimension of the output

# Training settings: batch size, maximum number of epochs
trBatchSize = 64
trMaxEpoch = 3

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


# In[ ]:


class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file, class_names, transform=None, policy="ones"):
        """
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels
        """
        image_names = []
        labels = []
        labels_string = []
        

        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k=0
            for line in csvReader:
                k+=1
                if k==100:
                    break
                image_name= line[0]
                label = line[5:]
                label_string=[]
                for i in range(14):
                    
                    #label_string[i] = class_names[i]
                    
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                            label_string.append(class_names[i])
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                                label_string.append(class_names[i])
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append('/home/dsv/maul8511/deep_learning_project/data/' + image_name)
                labels.append(label)
                labels_string.append(label_string)
                
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.labels_string = labels_string

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        label_string = self.labels_string[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        #return image, torch.FloatTensor(label), label_string

        return image, label, label_string

    def __len__(self):
        return len(self.image_names)


# In[ ]:


#TRANSFORM DATA

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
#transformList.append(transforms.Resize(imgtransCrop))
transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)


# In[ ]:


images = CheXpertDataSet(pathFileValid, class_names, transformSequence)


# ## Analysis

# Next, we will set up a list of analysis methods by preparing tuples containing the methods' string identifiers used by `innvestigate.analyzer.create_analyzer(...)`, some optional parameters, a post processing choice for visualizing the computed analysis and a title for the figure to render. Analyzers can be deactivated by simply commenting the corresponding lines, or added by creating a new tuple as below.
# 
# For a full list of methods refer to the dictionary `investigate.analyzer.analyzers`.
# 
# Note: Should you run into resource trouble, e.g. you are running that notebook on a computer without GPU or with only limited GPU memory, consider deactivating one or more analyzers by commenting the corresponding lines in the next cell.

# In[5]:



input_range = net["input_range"]

noise_scale = (input_range[1]-input_range[0]) * 0.1

# Methods we use and some properties.
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN                TITLE
    

    # Interaction
    
    #("deep_taylor.bounded",   {"low": input_range[0], "high": input_range[1]}, imgnetutils.heatmap,       "DeepTaylor"),
   
    ("lrp.z",                 {},                       imgnetutils.heatmap,       "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},           imgnetutils.heatmap,       "LRP-Epsilon"),
    ("lrp.sequential_preset_a_flat",{"epsilon": 1},     imgnetutils.heatmap,       "LRP-PresetAFlat"),
    ("lrp.sequential_preset_b_flat",{"epsilon": 1},     imgnetutils.heatmap,       "LRP-PresetBFlat"),
]


# The main loop below will now instantiate the analyzer objects based on the loaded/trained model and the analyzers' parameterizations above and compute the analyses.

# In[6]:


# Create model without trailing softmax
model_wo_softmax = model
#model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

# Create analyzers.
analyzers = []
for method in methods:
    try:
        analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                model_wo_softmax, # model without softmax output
                                                **method[1])      # optional analysis parameters
    except innvestigate.NotAnalyzeableModelException:
        # Not all methods work with all models.
        analyzer = None
    analyzers.append(analyzer)


# Now we analyze each image with the different analyzers:

# In[7]:


analysis = np.zeros([len(images), len(analyzers)]+net["image_shape"]+[3])
text = []

for i, (x, y, z) in enumerate(images):
    # Add batch axis.
    x = x[None, :, :, :].numpy()
    x_pp = imgnetutils.preprocess(x, net)

    # Predict final activations, probabilites, and label.
    presm = model_wo_softmax.predict_on_batch(x_pp)[0]
    prob = model.predict_on_batch(x_pp)[0]
    y_hat = prob.argmax()
    
    '''
    # Save prediction info:
    text.append(("%s" % label_to_class_name[y],    # ground truth label
                 "%.2f" % presm.max(),             # pre-softmax logits
                 "%.2f" % prob.max(),              # probabilistic softmax output  
                 "%s" % label_to_class_name[y_hat] # predicted label
                ))
    '''
    
    # Save prediction info:
    text.append(("%s" % z,    # ground truth label
                 "%.2f" % presm.max(),             # pre-softmax logits
                 "%.2f" % prob.max(),              # probabilistic softmax output  
                 "%s" % class_names[y_hat] # predicted label
                ))

    for aidx, analyzer in enumerate(analyzers):
        if methods[aidx][0] == "input":
            # Do not analyze, but keep not preprocessed input.
            a = x/255
        elif analyzer:
            # Analyze.
            a = analyzer.analyze(x_pp)

            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = imgnetutils.postprocess(a, color_conversion, channels_first)
            # Apply analysis postprocessing, e.g., creating a heatmap.
            a = methods[aidx][2](a)
        else:
            a = np.zeros_like(image)
        # Store the analysis.
        analysis[i, aidx] = a[0]


# Next, we visualize the analysis results:

# In[8]:


# Prepare the grid as rectengular list
grid = [[analysis[i, j] for j in range(analysis.shape[1])]
        for i in range(analysis.shape[0])]  
# Prepare the labels
label, presm, prob, pred = zip(*text)
row_labels_left = [('label: {}'.format(label[i]),'pred: {}'.format(pred[i])) for i in range(len(label))]
row_labels_right = [('logit: {}'.format(presm[i]),'prob: {}'.format(prob[i])) for i in range(len(label))]
col_labels = [''.join(method[3]) for method in methods]

# Plot the analysis.
eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                       file_name=os.environ.get("plot_file_name", None))


# This figure shows the analysis regarding the *actually predicted* class as computed by the selected analyzers. Each column shows the visualized results for different analyzers and each row shows the analyses wrt to one input sample. To the left of each row, the ground truth label `label` and the predicted label `pred` are show. To the right, the model's probabilistic (softmax) output is shown as `prob` and the logit output just before the terminating softmax layer as `logit`. Note that all analyses have been performed based on the logit output (layer).
# 
# 
# If you are curious about how **iNNvestigate** performs on *different* ImageNet model, have a look here: [Comparing networks on ImageNet](imagenet_network_comparison.ipynb)

