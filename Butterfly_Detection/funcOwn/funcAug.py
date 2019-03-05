#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:24:43 2018

@author: JieweiLu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

def data_augmentation(photo_filenames,i,index,ib,label,name,savePath):
    image = []
    graphMy = tf.Graph()
    with graphMy.as_default():
        image_data = tf.gfile.FastGFile(photo_filenames[i], 'rb').read()
        imageCrop = tf.image.decode_jpeg(image_data)
        
        image0 = imageCrop
        
        # translation
        image1 = tf.image.flip_up_down(imageCrop)
        image2 = tf.image.flip_left_right(imageCrop)
        image3 = tf.image.transpose_image(imageCrop)
        
        # random contrast 
        image4 = tf.image.random_contrast(imageCrop, 0.5,2)
    
        # adjust brightness
        image5 = tf.image.random_brightness(imageCrop,0.2)
    
        # adjust hue
        image6 = tf.image.random_hue(imageCrop,0.1)
    
        # adjust saturation
        image7 = tf.image.random_saturation(imageCrop,0.25,4)
    
        # rotate 20 degree and 160 degree
        image8 = tf.contrib.image.rotate(imageCrop,45*math.pi/180,interpolation='BILINEAR')
        image9 = tf.contrib.image.rotate(imageCrop,135*math.pi/180,interpolation='BILINEAR')
    
        # random crop
    
        # random scale
    
    
        image.append(image0)
        image.append(image1)
        image.append(image2)
        image.append(image3)
        image.append(image4)
        image.append(image5)
        image.append(image6)
        image.append(image7)
        image.append(image8)
        image.append(image9)
    
    sessMy = tf.Session(graph=graphMy)
    for j in range(len(image)):
        saveName = '%s_%s_%s_%s_%s' % (index,ib,label,j,name)
        encoded_image = tf.image.encode_jpeg(image[j])
        with tf.gfile.GFile(savePath+saveName,"wb") as f:
            f.write(sessMy.run(encoded_image))
    sessMy.close()