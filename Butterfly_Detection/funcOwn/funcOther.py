#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:10:31 2018

@author: JieweiLu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def show_image(image_):
    # show the original image
    _,ax = plt.subplots(figsize=(5,5))
    ax.imshow(image_) 
    plt.axis('off')
    

def getDictionInfor(diction):
    # extract the box and label information
    image_ = diction['image']
    bbox = diction['groundtruth_boxes']
    label = diction['groundtruth_classes']
    num_groundtruth_boxes = diction['num_groundtruth_boxes']
    filename = diction['filename']
    
        
    return image_,bbox,label,num_groundtruth_boxes,filename

def getCropImage(ib,image_,bbox):
    xmin = int(round(bbox[ib,1] * image_.shape[1]))
    xmax = int(round(bbox[ib,3] * image_.shape[1]))
    ymin = int(round(bbox[ib,0] * image_.shape[0]))
    ymax = int(round(bbox[ib,2] * image_.shape[0]))
    
    # convert the matrix to image
    imageObtain = Image.fromarray(image_.astype(np.uint8))
            
    # Crop the image
    box = (xmin,ymin,xmax,ymax)
    imageCrop = imageObtain.crop(box)
    
    return imageCrop

def saveImage(ib,i,filename,label,savePath,imageCrop,ia,dataName):
    # Save the image                      
    xu = '%s_%s_%s_%s_' % (i,ib+1,label[ib],ia) 
    
    # since the training dataset containing some complex characters
    if dataName == 'test':
        filenameStr = bytes.decode(filename)
        saveName = xu+filenameStr
    elif dataName == 'train':
        saveName = xu + 'IMG.jpg'
    elif dataName == 'kuo':
        saveName = xu + 'IMG.jpg'
    imageCrop.save(savePath+saveName)